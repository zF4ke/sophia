import { randomUUID } from "crypto";
import { getAppConfig } from "@/app/AppConfig";
import { SettingsService } from "@/app/SettingsService";
import { ProtectedChannelsService } from "@/app/ProtectedChannelsService";
import { ModelGateway, type ToolChatMessage } from "@/ai/ModelGateway";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import {
    answerConfidenceForInsufficient,
    classify,
    countEvidence,
    formatChannelContext,
} from "@/runtime/planning";
import { PromptRegistry } from "@/runtime/PromptRegistry";
import { TOOL_DEFINITIONS } from "@/runtime/toolSchemas";
import type {
    ApprovalRequest,
    BatchApprovalRequest,
    BatchedDestructiveItem,
    ChannelContextMessage,
    ConversationTurnSummary,
    EvidenceItem,
    RuntimeAnswer,
    RuntimeTraceEvent,
    StopReason,
    ToolArguments,
    ToolInvocationRecord,
    TurnInput,
} from "@/runtime/contracts";
import { getToolStrategy } from "@/runtime/tools";
import { type DiscordToolName, DISCORD_TOOL_NAMES, T } from "@/shared/discordTools";
import { isMutatingTool, describeApproval } from "@/tools/registry";
import type {
    DiscordToolResult,
    GroundedAnswerMode,
} from "@/shared/appTypes";

// ── Helpers: evidence extraction, identity formatting, retrieval session ──

function argsSignature(tool: string, args: ToolArguments) {
    return `${tool}:${JSON.stringify(args)}`;
}

function getRetrievalSummary(run: DiscordToolResult) {
    if (run.tool === "finish") return null;
    if (isMutatingTool(run.tool)) return null;
    const strategy = getToolStrategy(run.tool);
    return strategy.extractRetrievalSummary?.(run) ?? null;
}

function extractEvidence(
    run: DiscordToolResult
): EvidenceItem[] {
    if (run.tool === "finish") return [];
    if (isMutatingTool(run.tool)) return [];
    return getToolStrategy(run.tool).extractEvidence(run);
}

function isReusablePromptEvidence(item: EvidenceItem): boolean {
    return (
        item.evidenceRole === "message_evidence" ||
        item.evidenceRole === "semantic_evidence"
    );
}

function formatRecentToolRuns(runs: Array<{ toolName: string; summary: string; learned: string; outputJson: string }>): string {
    if (!runs.length) return "None.";
    return runs.map((r) => {
        const base = `- ${r.toolName}: ${r.summary}`;
        // For resolution/discovery tools, include the learned data (resolved IDs, structure)
        // so the model can reuse them directly without re-calling.
        if (
            (r.toolName === T.resolve_member_identity ||
             r.toolName === T.resolve_channel_targets ||
             r.toolName === T.list_guild_structure) &&
            r.learned && r.learned !== r.summary
        ) {
            return `${base}\n  Data: ${r.learned}`;
        }
        return base;
    }).join("\n");
}

function summarizeRecentTurns(turns: ConversationTurnSummary[]): string {
    if (!turns.length) {
        return "No recent conversation turns.";
    }

    return turns
        .map(
            (turn, index) =>
                `${index + 1}. user=${turn.question} | sophia=${turn.answer}`
        )
        .join("\n");
}

function sanitizeAnswer(answer: string | null | undefined): string {
    const normalized = (answer || "").trim();
    const banned = new Set([
        "I couldn't ground that in Discord evidence.",
        "I don't have enough Discord evidence to answer that yet.",
    ]);
    return banned.has(normalized) ? "" : normalized;
}

async function synthesizeAnswer(
    systemPrompt: string,
    question: string,
    toolHistory: ToolInvocationRecord[],
    evidence: EvidenceItem[],
    traceEvents?: RuntimeTraceEvent[]
): Promise<string> {
    const findingLines = toolHistory
        .filter((r) => !r.blocked && r.summary)
        .map((r) => `${r.tool}: ${r.summary}`)
        .join("\n");

    const evidenceLines = evidence
        .filter(isReusablePromptEvidence)
        .slice(0, 12)
        .map((e) => `${e.authorName || "unknown"}: ${e.content}`)
        .join("\n");

    const contextBlock = [
        findingLines ? `Tool findings:\n${findingLines}` : null,
        evidenceLines ? `Relevant messages:\n${evidenceLines}` : null,
    ].filter(Boolean).join("\n\n");

    const messages = [
        { role: "system" as const, content: systemPrompt },
        { role: "user" as const, content: question },
        ...(contextBlock
            ? [{ role: "assistant" as const, content: `Here is what I found:\n${contextBlock}` }]
            : []),
        {
            role: "user" as const,
            content: "Based on that research, answer the original question as concisely as possible. If you truly could not find it, say so in one sentence — do not ask for clarification or more details.",
        },
    ];

    try {
        const result = await ModelGateway.generateText(messages, {
            traceContext: { traceLabel: "synthesis_fallback", questionPreview: question, traceEvents: traceEvents ? [...traceEvents] : undefined },
        });
        return sanitizeAnswer(result);
    } catch {
        return "";
    }
}

// ── Constraints type (extracted from AppConfig.runtime) ──

interface RuntimeConstraints {
    maxToolCalls: number;
    maxLatencyBudgetMs: number;
    maxRepeatedCallSignature: number;
    maxPriorTurns: number;
    maxChannelMessages: number;
    maxToolRunsContext: number;
    maxEvidenceSlice: number;
}

// ── The tool names the model can call (excludes "finish") ──

const ALL_TOOL_NAMES = new Set<string>(DISCORD_TOOL_NAMES);

function isKnownTool(name: string): name is DiscordToolName {
    return ALL_TOOL_NAMES.has(name);
}

// ── Max chars for a single tool result injected back into the conversation ──
const MAX_TOOL_RESULT_CHARS = 150_000;

function truncateToolResult(json: string): string {
    if (json.length <= MAX_TOOL_RESULT_CHARS) return json;
    return json.slice(0, MAX_TOOL_RESULT_CHARS) + "\n...[truncated]";
}

// ── Context overflow detection and pruning ──
const CONTEXT_HEADROOM_RATIO = 0.80;

function isContextOverflow(promptTokens: number, contextWindow: number): boolean {
    return promptTokens >= contextWindow * CONTEXT_HEADROOM_RATIO;
}

function pruneOldToolOutputs(messages: ToolChatMessage[]): number {
    let pruned = 0;
    // Keep the system prompt (index 0), the user message (index 1),
    // and the last 4 messages (latest tool interaction). Prune everything in between.
    const protectedTail = 4;
    const lastPrunableIndex = messages.length - protectedTail;

    for (let i = 2; i < lastPrunableIndex; i++) {
        const msg = messages[i];
        if (msg.role === "tool" && msg.content.length > 200) {
            (msg as { content: string }).content = JSON.stringify({ pruned: true, note: "Earlier tool output pruned to save context space." });
            pruned += 1;
        }
    }
    return pruned;
}

// ── Runtime ──

export class Runtime {
    public static async answer(input: TurnInput): Promise<RuntimeAnswer> {
        const config = getAppConfig();
        const requestId = randomUUID();
        const threadId = input.conversation.key;
        const guildId = input.guild?.id || null;
        const channelId = input.currentChannelId || null;
        const actorId = input.user.id;

        const constraints: RuntimeConstraints = {
            maxToolCalls: config.runtime.maxToolCalls,
            maxLatencyBudgetMs: config.runtime.maxLatencyBudgetMs,
            maxRepeatedCallSignature: config.runtime.maxRepeatedCallSignature,
            maxPriorTurns: config.runtime.maxPriorTurns,
            maxChannelMessages: config.runtime.maxChannelMessages,
            maxToolRunsContext: config.runtime.maxToolRunsContext,
            maxEvidenceSlice: config.runtime.maxEvidenceSlice,
        };
        const settings = SettingsService.load();

        const traceEvents: RuntimeTraceEvent[] = [];
        const toolHistory: ToolInvocationRecord[] = [];
        const evidence: EvidenceItem[] = [];
        let confidence: GroundedAnswerMode = "insufficient";
        let stopReason: StopReason | null = null;

        const trace = (label: string, detail: string) => {
            traceEvents.push({ label, detail, timestamp: Date.now() });
        };

        try {
            // ── 1. Debug session setup ──
            await input.debugSession?.setClassifying();
            await input.debugSession?.setRequesterContext?.(input.requesterDisplayName, input.trigger);
            await input.debugSession?.setConversationContext?.({
                threadId,
                kind: input.conversation.kind,
                replyAnchorMessageId: input.conversation.replyAnchorMessageId,
                replyContext: input.replyContext,
            });

            trace("ingest_turn", `trigger=${input.trigger}; conversation=${input.conversation.kind}; requester=${input.requesterDisplayName}`);

            // ── 2. Load memory (recent turns, channel context, prior evidence) ──
            const recentTurns = await DiscordMemoryService.getRecentRuntimeRunsAsync(
                threadId,
                constraints.maxPriorTurns
            );
            const recentToolRuns = await DiscordMemoryService.getRecentToolRunsAsync(
                threadId,
                constraints.maxToolRunsContext
            );
            const channelMessages = channelId
                ? await DiscordMemoryService.getRecentChannelMessagesAsync(channelId, constraints.maxChannelMessages)
                : [];
            const channelContext: ChannelContextMessage[] = channelMessages.map((msg) => ({
                authorName: msg.authorName,
                content: msg.content.slice(0, 150),
                createdTimestamp: msg.createdTimestamp,
            }));

            // Reconstruct prior evidence from recent tool runs
            const seenEvidence = new Set<string>();
            for (const run of [...recentToolRuns].reverse()) {
                try {
                    const parsedOutput = JSON.parse(run.outputJson) as DiscordToolResult;
                    const evidenceItems = extractEvidence(
                        parsedOutput
                    );
                    for (const item of evidenceItems) {
                        const key = [
                            item.tool,
                            item.jumpLink || "",
                            item.channelId || "",
                            item.authorId || "",
                            item.createdTimestamp || "",
                            item.content,
                        ].join("::");
                        if (!seenEvidence.has(key)) {
                            seenEvidence.add(key);
                            evidence.push(item);
                        }
                    }
                } catch {
                    continue;
                }
            }
            // Trim to budget
            while (evidence.length > constraints.maxEvidenceSlice) {
                evidence.shift();
            }

            trace("load_memory", `Loaded ${recentTurns.length} prior turn(s), ${channelContext.length} channel msg(s), reused ${evidence.length} evidence item(s).`);

            // ── 3. Build system prompt ──
            const promptEvidence = evidence.filter(isReusablePromptEvidence);
            const priorEvidenceSummary = promptEvidence.length
                ? promptEvidence.map((e) => {
                      const who = e.authorName || "?";
                      const where = e.channelName ? `#${e.channelName}` : "";
                      return `${who}${where ? ` in ${where}` : ""}: ${e.content}`;
                  }).join("\n")
                : "None.";

            const personality = SettingsService.load().personality;
            let personalityOverride = "";
            if (personality === "classic") {
                try {
                    personalityOverride = PromptRegistry.load("system/personality_classic_override");
                } catch {
                    personalityOverride = "";
                }
            }

            const systemPrompt = PromptRegistry.render("runtime/agent_loop", {
                guild_name: input.guild?.name || "DM",
                guild_id: guildId || "none",
                channel_name: channelId ? `<#${channelId}>` : "DM",
                channel_id: channelId || "none",
                requester_display_name: input.requesterDisplayName,
                actor_id: actorId,
                current_date: new Date().toISOString().slice(0, 10),
                trigger: input.trigger,
                recent_turns: summarizeRecentTurns(recentTurns),
                channel_context: formatChannelContext(channelContext),
                prior_evidence: priorEvidenceSummary,
                tool_context: formatRecentToolRuns(recentToolRuns),
                reply_context: input.replyContext
                    ? `Replying to ${input.replyContext.authorDisplayName}: "${input.replyContext.content}"`
                    : "Not a reply.",
                max_tool_calls: String(constraints.maxToolCalls),
                personality_override: personalityOverride,
            });

            // ── 4. Agent loop ──
            const messages: ToolChatMessage[] = [
                { role: "system", content: systemPrompt },
                { role: "user", content: input.question },
            ];

            const startedAt = Date.now();
            let pausedLatencyMs = 0;
            const repeated = new Map<string, number>();
            let answer = "";
            let totalToolCalls = 0;
            let cumulativePromptTokens = 0;
            let cumulativeCompletionTokens = 0;

            const getActiveElapsedMs = () =>
                Math.max(0, Date.now() - startedAt - pausedLatencyMs);

            await input.debugSession?.setClassification("discord_grounded");

            for (let iteration = 0; iteration < constraints.maxToolCalls + 1; iteration += 1) {
                // Latency guard
                if (getActiveElapsedMs() >= constraints.maxLatencyBudgetMs) {
                    stopReason = "budget_exhausted";
                    trace("stop", "Reached the latency budget.");
                    break;
                }

                await input.debugSession?.setPlanning(iteration + 1);
                const result = await ModelGateway.generateWithTools(messages, {
                    tools: TOOL_DEFINITIONS,
                    traceContext: {
                        traceLabel: `agent_loop_iter_${iteration}`,
                        questionPreview: input.question,
                        traceEvents: [...traceEvents],
                    },
                });

                // Track token usage
                if (result.usage) {
                    cumulativePromptTokens += result.usage.promptTokens;
                    cumulativeCompletionTokens += result.usage.completionTokens;
                }

                // Context overflow guard — prune old tool outputs if approaching limit
                if (result.usage && isContextOverflow(result.usage.promptTokens, config.modelProfile.contextWindow)) {
                    const pruned = pruneOldToolOutputs(messages);
                    if (pruned > 0) {
                        trace("context_prune", `Pruned ${pruned} old tool outputs (prompt tokens: ${result.usage.promptTokens}/${config.modelProfile.contextWindow}).`);
                    }
                }

                // No tool calls → model produced a text response (shouldn't happen with finish tool, but handle it)
                if (result.toolCalls.length === 0) {
                    if (result.content) {
                        answer = result.content;
                        stopReason = totalToolCalls > 0 ? "evidence_sufficient" : "direct_answer";
                    } else {
                        stopReason = "no_useful_next_step";
                    }
                    trace("model_response", `No tool calls. finishReason=${result.finishReason}`);
                    break;
                }

                // Process each tool call in the response
                const assistantMessage: ToolChatMessage = {
                    role: "assistant",
                    content: result.content,
                    tool_calls: result.toolCalls,
                };
                messages.push(assistantMessage);

                let loopDone = false;

                // Helper: execute a tool and record results
                const executeToolCall = async (
                    tc: typeof result.toolCalls[number],
                    toolName: DiscordToolName,
                    parsedArgs: ToolArguments,
                ) => {
                    const capability = CapabilityRegistry.get(toolName);

                    await input.debugSession?.setToolRunning(toolName, []);
                    const toolStartedAt = Date.now();
                    const output = await capability.run(
                        {
                            guild: input.guild || null,
                            question: input.question,
                            currentChannelId: channelId,
                            onProgress: async (tn, summary) => {
                                trace("tool_progress", `${tn}: ${summary}`);
                                await input.debugSession?.setToolProgress?.(tn, summary);
                            },
                        },
                        parsedArgs,
                    );
                    const toolDurationMs = Date.now() - toolStartedAt;
                    totalToolCalls += 1;

                    const evidenceItems = extractEvidence(output);
                    const learned = evidenceItems.map((item) => item.content).join(" | ") || output.summary;
                    const retrieval = getRetrievalSummary(output);

                    const record: ToolInvocationRecord = {
                        tool: toolName,
                        arguments: parsedArgs,
                        summary: output.summary,
                        learned,
                        confidenceImproved: evidenceItems.some((item) => item.strength !== "weak"),
                        output,
                        durationMs: toolDurationMs,
                        retrievalSummary: retrieval,
                    };
                    toolHistory.push(record);
                    evidence.push(...evidenceItems);

                    trace("tool_result", `${toolName}: ${output.summary}`);

                    const resultPayload = output.errorMessage
                        ? JSON.stringify({ error: output.errorMessage })
                        : JSON.stringify({ summary: output.summary, data: output.data });
                    messages.push({
                        role: "tool",
                        tool_call_id: tc.id,
                        content: truncateToolResult(resultPayload),
                    });

                    await DiscordMemoryService.recordToolRun(
                        requestId,
                        guildId,
                        channelId,
                        actorId,
                        input.question,
                        toolName,
                        JSON.stringify(parsedArgs),
                        output.summary,
                        learned,
                        JSON.stringify(output),
                        record.confidenceImproved,
                        toolDurationMs,
                    );

                    await input.debugSession?.setToolResult(toolName, output.summary);
                    await input.debugSession?.setRetrievalSummary?.(retrieval || {
                        mode: "history",
                        cacheHit: false,
                        liveEscalated: false,
                        searchedChannelIds: [],
                        fetchedChannelIds: [],
                        cacheEnriched: false,
                        evidenceSufficient: false,
                        strongResultCount: 0,
                        weakResultCount: 0,
                        historyMessageCount: 0,
                        semanticMatchCount: 0,
                        accumulatedUniqueCount: 0,
                        sourceOrigin: "none",
                        continuationAvailable: false,
                        historyContinuationAvailable: false,
                        historyCursorByChannel: {},
                        semanticContinuationAvailable: false,
                        semanticCursor: null,
                        exhaustedChannelIds: [],
                        historyExhausted: false,
                        semanticExhausted: false,
                        beforeTimestamp: null,
                        afterTimestamp: null,
                        activeChannelIds: [],
                    });
                };

                const getProtectedTargetChannelId = (
                    toolName: DiscordToolName,
                    parsedArgs: ToolArguments,
                ): string | null => {
                    const capability = CapabilityRegistry.get(toolName);
                    if (capability.sideEffectLevel !== "destructive") return null;

                    const targetChannelId = parsedArgs.channel_id
                        ? String(parsedArgs.channel_id)
                        : null;

                    if (!targetChannelId || !ProtectedChannelsService.isProtected(targetChannelId)) {
                        return null;
                    }

                    return targetChannelId;
                };

                const recordProtectedChannelBlock = async (
                    tc: typeof result.toolCalls[number],
                    toolName: DiscordToolName,
                    parsedArgs: ToolArguments,
                    targetChannelId: string,
                ) => {
                    const denyMsg = `Action NOT executed. Auto-blocked because channel ${targetChannelId} is protected. Do not tell the user this action was completed.`;
                    // console.log(`[ProtectedChannels] AUTO-BLOCKED ${toolName} on channel ${targetChannelId}`);

                    // Send visual notification card (best-effort)
                    if (input.protectedBlockNotifier) {
                        const approvalDescription = describeApproval(toolName, parsedArgs);
                        await input.protectedBlockNotifier({
                            requestId: `${requestId}:${toolName}:${totalToolCalls}:protected`,
                            toolName,
                            toolArgs: parsedArgs,
                            description: approvalDescription,
                            sideEffectLevel: "destructive",
                            requesterId: actorId,
                        }).catch(() => { /* best-effort */ });
                    }

                    messages.push({
                        role: "tool",
                        tool_call_id: tc.id,
                        content: JSON.stringify({ error: denyMsg }),
                    });
                    toolHistory.push({
                        tool: toolName,
                        arguments: parsedArgs,
                        summary: `Blocked: protected channel ${targetChannelId}.`,
                        learned: denyMsg,
                        confidenceImproved: false,
                        output: {
                            tool: toolName,
                            summary: `Blocked: <#${targetChannelId}> is a protected channel.`,
                            data: null,
                            errorMessage: `Channel ${targetChannelId} is protected and cannot be modified or deleted.`,
                        },
                        durationMs: 0,
                        blocked: true,
                    });
                    trace("tool_blocked", `${toolName} auto-blocked on protected channel ${targetChannelId} after approval.`);
                };

                // Collect destructive tool calls for batch approval
                interface QueuedDestructiveCall {
                    tc: typeof result.toolCalls[number];
                    toolName: DiscordToolName;
                    parsedArgs: ToolArguments;
                    description: string;
                }
                const destructiveBatch: QueuedDestructiveCall[] = [];

                for (const tc of result.toolCalls) {
                    const toolName = tc.function.name;
                    let parsedArgs: ToolArguments;
                    try {
                        parsedArgs = JSON.parse(tc.function.arguments || "{}") as ToolArguments;
                    } catch {
                        parsedArgs = {};
                    }

                    // ── Handle "finish" tool ──
                    if (toolName === "finish") {
                        answer = (parsedArgs.answer as string) || result.content || "";
                        stopReason = totalToolCalls > 0 ? "evidence_sufficient" : "direct_answer";
                        trace("finish", `Model called finish.`);
                        messages.push({
                            role: "tool",
                            tool_call_id: tc.id,
                            content: JSON.stringify({ ok: true }),
                        });
                        loopDone = true;
                        break;
                    }

                    // ── Handle known tool calls ──
                    if (!isKnownTool(toolName)) {
                        messages.push({
                            role: "tool",
                            tool_call_id: tc.id,
                            content: JSON.stringify({ error: `Unknown tool: ${toolName}` }),
                        });
                        trace("tool_error", `Unknown tool ${toolName}`);
                        continue;
                    }

                    // Budget guard
                    if (totalToolCalls >= constraints.maxToolCalls) {
                        messages.push({
                            role: "tool",
                            tool_call_id: tc.id,
                            content: JSON.stringify({ error: "Tool call budget exhausted. Call finish now." }),
                        });
                        stopReason = "budget_exhausted";
                        trace("stop", "Tool-call budget exhausted.");
                        continue;
                    }

                    // Repeated-call guard
                    const signature = argsSignature(toolName, parsedArgs);
                    const seen = (repeated.get(signature) || 0) + 1;
                    repeated.set(signature, seen);
                    if (seen > constraints.maxRepeatedCallSignature) {
                        const blockMsg = `You already called ${toolName} with these exact arguments. Use different arguments or a different tool.`;
                        messages.push({
                            role: "tool",
                            tool_call_id: tc.id,
                            content: JSON.stringify({ error: blockMsg }),
                        });
                        const warningRecord: ToolInvocationRecord = {
                            tool: toolName,
                            arguments: parsedArgs,
                            summary: "Blocked: repeated call.",
                            learned: blockMsg,
                            confidenceImproved: false,
                            output: { tool: toolName, summary: "Repeated call blocked.", data: null, errorMessage: blockMsg },
                            durationMs: 0,
                            blocked: true,
                        };
                        toolHistory.push(warningRecord);
                        trace("step", `Blocked repeated call ${signature}.`);

                        const repeatedViolations = [...repeated.values()].filter((v) => v > constraints.maxRepeatedCallSignature).length;
                        if (repeatedViolations >= 2) {
                            stopReason = "confidence_plateau";
                            loopDone = true;
                            break;
                        }
                        continue;
                    }

                    // ── Approval gate for write/destructive tools ──
                    const capability = CapabilityRegistry.get(toolName);
                    if (capability.sideEffectLevel !== "none") {
                        const autoApproveWrite =
                            capability.sideEffectLevel === "write" &&
                            settings.runtime.autoApproveWrites;
                        if (autoApproveWrite) {
                            trace("approval_auto_approved", `${toolName}: auto-approved by setting (write-only).`);
                        }

                        // ── Destructive → auto-block if protected, otherwise queue for batch approval ──
                        if (!autoApproveWrite && capability.sideEffectLevel === "destructive") {
                            const protectedId = getProtectedTargetChannelId(toolName, parsedArgs);
                            if (protectedId) {
                                await recordProtectedChannelBlock(tc, toolName, parsedArgs, protectedId);
                                continue;
                            }
                            const approvalDescription = describeApproval(toolName, parsedArgs);
                            destructiveBatch.push({ tc, toolName, parsedArgs, description: approvalDescription });
                            trace("batch_queued", `${toolName}: queued for batch approval — ${approvalDescription}`);
                            continue; // skip execution; will be resolved after batch approval
                        }

                        // ── Write (non-auto-approved) → individual approval ──
                        if (!autoApproveWrite) {
                        const approvalDescription = describeApproval(toolName, parsedArgs);
                        const approvalRequest: ApprovalRequest = {
                            requestId: `${requestId}:${toolName}:${totalToolCalls}`,
                            toolName,
                            toolArgs: parsedArgs,
                            description: approvalDescription,
                            sideEffectLevel: capability.sideEffectLevel,
                            requesterId: actorId,
                        };
                        trace("approval_requested", `${toolName}: ${approvalDescription}`);
                        await input.debugSession?.setToolRunning(toolName, ["⏳ Awaiting admin approval"]);

                        if (!input.approvalGate) {
                            const denyMsg = "Write operations require admin approval, but no approval channel is available.";
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ error: denyMsg }),
                            });
                            trace("approval_denied", `${toolName}: auto-denied (no gate)`);
                            continue;
                        }

                        // Stop typing while waiting for human approval
                        await input.activityIndicator?.stop();

                        const approvalWaitStartedAt = Date.now();
                        const approvalResult = await input.approvalGate(approvalRequest);
                        const approvalWaitMs = Math.max(
                            0,
                            Date.now() - approvalWaitStartedAt,
                        );
                        pausedLatencyMs += approvalWaitMs;
                        trace(
                            "approval_wait",
                            `${toolName}: waited ${approvalWaitMs}ms for admin approval (excluded from latency budget).`,
                        );
                        if (approvalResult.haltExecution) {
                            const stopMsg = `Execution stopped by ${approvalResult.decidedBy}.`;
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ error: stopMsg }),
                            });
                            const stoppedRecord: ToolInvocationRecord = {
                                tool: toolName,
                                arguments: parsedArgs,
                                summary: `Execution stopped by ${approvalResult.decidedBy}.`,
                                learned: stopMsg,
                                confidenceImproved: false,
                                output: { tool: toolName, summary: stopMsg, data: null, errorMessage: stopMsg },
                                durationMs: 0,
                                blocked: true,
                            };
                            toolHistory.push(stoppedRecord);
                            trace("stop", `${toolName}: execution stopped by ${approvalResult.decidedBy}`);
                            stopReason = "execution_stopped_by_admin";
                            loopDone = true;
                            await input.activityIndicator?.startThinking();
                            break;
                        }
                        if (!approvalResult.approved) {
                            const correctionNote = approvalResult.correction
                                ? ` Admin correction: "${approvalResult.correction}". Adjust your approach based on this feedback.`
                                : "";
                            const denyMsg = `Action NOT executed. Denied by ${approvalResult.decidedBy}.${correctionNote} Do not tell the user this action was completed.`;
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ error: denyMsg }),
                            });
                            const deniedRecord: ToolInvocationRecord = {
                                tool: toolName,
                                arguments: parsedArgs,
                                summary: `Denied by ${approvalResult.decidedBy}.${correctionNote}`,
                                learned: denyMsg,
                                confidenceImproved: false,
                                output: { tool: toolName, summary: denyMsg, data: null, errorMessage: denyMsg },
                                durationMs: 0,
                                blocked: true,
                            };
                            toolHistory.push(deniedRecord);
                            trace("approval_denied", `${toolName}: denied by ${approvalResult.decidedBy}${approvalResult.correction ? ` (correction: ${approvalResult.correction})` : ""}`);
                            await input.activityIndicator?.startThinking();
                            if (approvalResult.decidedBy === "timeout") {
                                loopDone = true;
                            }
                            continue;
                        }
                        trace("approval_granted", `${toolName}: approved by ${approvalResult.decidedBy}`);
                        await input.activityIndicator?.startThinking();
                        }
                    }

                    // Execute the capability
                    await executeToolCall(tc, toolName, parsedArgs);
                }

                // ── Batch destructive approval (after processing all tool calls in this iteration) ──
                if (destructiveBatch.length > 0 && !loopDone) {
                    if (destructiveBatch.length === 1 && input.approvalGate) {
                        // Single destructive action — use the single-item approval gate (no batch card)
                        const item = destructiveBatch[0];
                        const approvalRequest: ApprovalRequest = {
                            requestId: `${requestId}:${item.toolName}:${totalToolCalls}`,
                            toolName: item.toolName,
                            toolArgs: item.parsedArgs,
                            description: item.description,
                            sideEffectLevel: "destructive",
                            requesterId: actorId,
                        };
                        trace("approval_requested", `${item.toolName}: ${item.description}`);
                        await input.debugSession?.setToolRunning(item.toolName, ["⏳ Awaiting admin approval"]);
                        await input.activityIndicator?.stop();

                        const singleDestructiveWaitStartedAt = Date.now();
                        const approvalResult = await input.approvalGate(approvalRequest);
                        const singleDestructiveWaitMs = Math.max(0, Date.now() - singleDestructiveWaitStartedAt);
                        pausedLatencyMs += singleDestructiveWaitMs;
                        trace("approval_wait", `${item.toolName}: waited ${singleDestructiveWaitMs}ms for admin approval (excluded from latency budget).`);

                        if (approvalResult.haltExecution) {
                            const stopMsg = `Execution stopped by ${approvalResult.decidedBy}.`;
                            messages.push({ role: "tool", tool_call_id: item.tc.id, content: JSON.stringify({ error: stopMsg }) });
                            const stoppedRecord: ToolInvocationRecord = {
                                tool: item.toolName,
                                arguments: item.parsedArgs,
                                summary: `Execution stopped by ${approvalResult.decidedBy}.`,
                                learned: stopMsg,
                                confidenceImproved: false,
                                output: { tool: item.toolName, summary: stopMsg, data: null, errorMessage: stopMsg },
                                durationMs: 0,
                                blocked: true,
                            };
                            toolHistory.push(stoppedRecord);
                            trace("stop", `${item.toolName}: execution stopped by ${approvalResult.decidedBy}`);
                            stopReason = "execution_stopped_by_admin";
                            loopDone = true;
                            await input.activityIndicator?.startThinking();
                        } else if (!approvalResult.approved) {
                            const correctionNote = approvalResult.correction
                                ? ` Admin correction: "${approvalResult.correction}". Adjust your approach based on this feedback.`
                                : "";
                            const denyMsg = `Action NOT executed. Denied by ${approvalResult.decidedBy}.${correctionNote} Do not tell the user this action was completed.`;
                            messages.push({ role: "tool", tool_call_id: item.tc.id, content: JSON.stringify({ error: denyMsg }) });
                            const deniedRecord: ToolInvocationRecord = {
                                tool: item.toolName,
                                arguments: item.parsedArgs,
                                summary: `Denied by ${approvalResult.decidedBy}.${correctionNote}`,
                                learned: denyMsg,
                                confidenceImproved: false,
                                output: { tool: item.toolName, summary: denyMsg, data: null, errorMessage: denyMsg },
                                durationMs: 0,
                                blocked: true,
                            };
                            toolHistory.push(deniedRecord);
                            trace("approval_denied", `${item.toolName}: denied by ${approvalResult.decidedBy}${approvalResult.correction ? ` (correction: ${approvalResult.correction})` : ""}`);
                            await input.activityIndicator?.startThinking();
                            if (approvalResult.decidedBy === "timeout") {
                                loopDone = true;
                            }
                        } else {
                            trace("approval_granted", `${item.toolName}: approved by ${approvalResult.decidedBy}`);
                            await input.activityIndicator?.startThinking();
                            const protectedChannelId = getProtectedTargetChannelId(item.toolName, item.parsedArgs);
                            if (protectedChannelId) {
                                await recordProtectedChannelBlock(item.tc, item.toolName, item.parsedArgs, protectedChannelId);
                            } else {
                                await executeToolCall(item.tc, item.toolName, item.parsedArgs);
                            }
                        }
                    } else if (!input.batchApprovalGate) {
                        // No batch gate — auto-deny all queued destructive tools
                        for (const item of destructiveBatch) {
                            const denyMsg = "Destructive operations require admin approval, but no approval channel is available.";
                            messages.push({
                                role: "tool",
                                tool_call_id: item.tc.id,
                                content: JSON.stringify({ error: denyMsg }),
                            });
                            trace("approval_denied", `${item.toolName}: auto-denied (no gate)`);
                        }
                    } else {
                        const batchId = `${requestId}:batch:${iteration}`;
                        const batchItems: BatchedDestructiveItem[] = destructiveBatch.map((item) => {
                            // Resolve the Discord Category the target channel belongs to
                            const channelId = (item.parsedArgs.channel_id as string) ?? null;
                            let targetCategory: BatchedDestructiveItem["targetCategory"] = null;
                            if (channelId && input.guild?.channels) {
                                const ch = input.guild.channels.cache.get(channelId);
                                if (ch?.parent) {
                                    targetCategory = { id: ch.parent.id, name: ch.parent.name };
                                }
                            }
                            return {
                                toolCallId: item.tc.id,
                                toolName: item.toolName,
                                toolArgs: item.parsedArgs,
                                description: item.description,
                                targetCategory,
                            };
                        });
                        const batchRequest: BatchApprovalRequest = {
                            batchId,
                            items: batchItems,
                            requesterId: actorId,
                        };
                        trace("batch_approval_requested", `${destructiveBatch.length} destructive action(s) pending batch approval.`);
                        await input.debugSession?.setToolRunning("batch_approval", [`⏳ Awaiting batch approval (${destructiveBatch.length} actions)`]);
                        await input.activityIndicator?.stop();

                        const batchWaitStartedAt = Date.now();
                        const batchResult = await input.batchApprovalGate(batchRequest);
                        const batchWaitMs = Math.max(0, Date.now() - batchWaitStartedAt);
                        pausedLatencyMs += batchWaitMs;
                        trace("batch_approval_wait", `Waited ${batchWaitMs}ms for batch approval (excluded from latency budget).`);

                        if (batchResult.haltExecution) {
                            for (const item of destructiveBatch) {
                                const stopMsg = `Execution stopped by ${batchResult.decidedBy}.`;
                                messages.push({
                                    role: "tool",
                                    tool_call_id: item.tc.id,
                                    content: JSON.stringify({ error: stopMsg }),
                                });
                                const stoppedRecord: ToolInvocationRecord = {
                                    tool: item.toolName,
                                    arguments: item.parsedArgs,
                                    summary: stopMsg,
                                    learned: stopMsg,
                                    confidenceImproved: false,
                                    output: { tool: item.toolName, summary: stopMsg, data: null, errorMessage: stopMsg },
                                    durationMs: 0,
                                    blocked: true,
                                };
                                toolHistory.push(stoppedRecord);
                            }
                            stopReason = "execution_stopped_by_admin";
                            loopDone = true;
                            trace("stop", `Batch execution stopped by ${batchResult.decidedBy}`);
                            await input.activityIndicator?.startThinking();
                        } else {
                            await input.activityIndicator?.startThinking();
                            const batchCorrectionNote = batchResult.correction
                                ? ` Admin correction: "${batchResult.correction}". Adjust your approach based on this feedback.`
                                : "";

                            if (batchResult.decidedBy === "timeout") {
                                // No admin present (timeout) — stop the loop so the model can't queue
                                // the same destructive actions again in the next iteration.
                                loopDone = true;
                            }

                            for (const item of destructiveBatch) {
                                const decision = batchResult.decisions[item.tc.id];
                                if (decision === "approved") {
                                    trace("batch_item_approved", `${item.toolName}: approved in batch by ${batchResult.decidedBy}`);
                                    const protectedChannelId = getProtectedTargetChannelId(item.toolName, item.parsedArgs);
                                    if (protectedChannelId) {
                                        await recordProtectedChannelBlock(item.tc, item.toolName, item.parsedArgs, protectedChannelId);
                                    } else {
                                        await executeToolCall(item.tc, item.toolName, item.parsedArgs);
                                    }
                                } else {
                                    const denyMsg = `Action NOT executed. Denied by ${batchResult.decidedBy}.${batchCorrectionNote} Do not tell the user this action was completed.`;
                                    messages.push({
                                        role: "tool",
                                        tool_call_id: item.tc.id,
                                        content: JSON.stringify({ error: denyMsg }),
                                    });
                                    const deniedRecord: ToolInvocationRecord = {
                                        tool: item.toolName,
                                        arguments: item.parsedArgs,
                                        summary: `Denied by ${batchResult.decidedBy}.${batchCorrectionNote}`,
                                        learned: denyMsg,
                                        confidenceImproved: false,
                                        output: { tool: item.toolName, summary: denyMsg, data: null, errorMessage: denyMsg },
                                        durationMs: 0,
                                        blocked: true,
                                    };
                                    toolHistory.push(deniedRecord);
                                    trace("batch_item_denied", `${item.toolName}: denied in batch by ${batchResult.decidedBy}${batchResult.correction ? ` (correction: ${batchResult.correction})` : ""}`);
                                }
                            }
                        }
                    }
                }

                if (loopDone) break;
            }

            // ── 5. Determine final answer ──
            if (!answer) {
                // Model never called finish — synthesize from what was found
                if (!stopReason) stopReason = "insufficient_evidence";
                confidence = toolHistory.length > 0
                    ? answerConfidenceForInsufficient({ evidence })
                    : "insufficient";
                answer = toolHistory.length > 0
                    ? await synthesizeAnswer(systemPrompt, input.question, toolHistory, evidence, traceEvents)
                    : "";
            } else {
                answer = sanitizeAnswer(answer);
                if (!stopReason) stopReason = "direct_answer";
                confidence = toolHistory.length > 0 ? "confident" : "best_effort";
            }

            const runtimeMode = toolHistory.length > 0 ? "research" : "conversation";
            const classification = classify(runtimeMode);

            // ── 6. Persist run ──
            await DiscordMemoryService.recordRuntimeRun({
                requestId,
                threadId,
                guildId,
                channelId,
                actorId,
                trigger: input.trigger,
                classificationMode: classification.mode,
                runtimeMode,
                stopReason: stopReason || "direct_answer",
                confidence,
                question: input.question,
                answer,
                traceEvents,
            });
            trace("persist_run", "Saved runtime run and trace events.");

            // ── 7. Final debug session updates ──
            const summary = countEvidence({ evidence });
            await input.debugSession?.setRuntimeMode?.(runtimeMode);
            await input.debugSession?.setEvidenceSummary(
                {
                    messageEvidenceCount: summary.messageEvidenceCount,
                    liveEvidenceCount: summary.liveEvidenceCount,
                    sufficient: stopReason === "evidence_sufficient",
                },
                "agent_loop",
                confidence
            );
            await input.debugSession?.setStopReason?.(
                stopReason || "direct_answer",
                deriveStopDetail(stopReason || "direct_answer", traceEvents)
            );
            await input.debugSession?.setConfidence?.(confidence);
            const contextUsagePercent = cumulativePromptTokens > 0 && config.modelProfile.contextWindow > 0
                ? (cumulativePromptTokens / config.modelProfile.contextWindow) * 100
                : null;
            await input.debugSession?.setTokenUsage?.(cumulativePromptTokens, cumulativeCompletionTokens, contextUsagePercent);
            const REALTIME_TRACE_LABELS = new Set(["tool_result", "tool_call", "tool_progress"]);
            for (const event of traceEvents.slice(-8)) {
                if (REALTIME_TRACE_LABELS.has(event.label)) continue;
                await input.debugSession?.setTraceEvent?.(event.label, event.detail, event.timestamp);
            }
            await input.debugSession?.setGenerating();
            await input.debugSession?.finishSuccess(stopReason || "completed");

            return {
                requestId,
                threadId,
                answer,
                citations: collectCitations(toolHistory),
                classification,
                toolRuns: toolHistory.filter((item) => !item.blocked).map((item) => item.output),
                confidence,
            };
        } catch (error) {
            const errorDetail = error instanceof Error
                ? { message: error.message, stack: error.stack, ...(error as any) }
                : error;
            const safeErrorDetail = (() => {
                try {
                    return JSON.parse(JSON.stringify(errorDetail, (_k, v) => {
                        if (v instanceof Headers) return Object.fromEntries(v.entries());
                        return v;
                    }));
                } catch {
                    return String(error);
                }
            })();
            console.error("[Runtime] Unhandled error — returning conversational fallback.", {
                requestId,
                question: input.question,
                actorId,
                guildId,
                channelId,
                model: config.modelProfile.chatModel,
                error: safeErrorDetail,
            });
            await input.debugSession?.setTraceEvent?.(
                "runtime_error_fallback",
                "The runtime hit an internal error and returned a conversational fallback instead."
            );
            await input.debugSession?.finishError(error);
            return {
                requestId,
                threadId,
                answer: "",
                citations: [],
                classification: classify("conversation"),
                toolRuns: [],
                confidence: "best_effort",
            };
        }
    }
}

function deriveStopDetail(
    stopReason: StopReason | null | undefined,
    traceEvents: Array<{ label: string; detail: string }>
): string | null {
    if (!stopReason) {
        return null;
    }

    const findLast = (predicate: (event: { label: string; detail: string }) => boolean) => {
        for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
            const event = traceEvents[index];
            if (predicate(event)) {
                return event.detail;
            }
        }
        return null;
    };

    if (stopReason === "budget_exhausted") {
        return findLast((event) => event.label === "stop");
    }

    if (stopReason === "confidence_plateau") {
        return findLast(
            (event) => event.label === "step" && event.detail.startsWith("Blocked repeated call ")
        );
    }

    if (stopReason === "execution_stopped_by_admin") {
        return findLast((event) => event.label === "stop");
    }

    if (stopReason === "no_useful_next_step") {
        return findLast((event) => event.label === "step");
    }

    if (stopReason === "evidence_sufficient" || stopReason === "insufficient_evidence") {
        return findLast((event) => event.label === "finish");
    }

    if (stopReason === "direct_answer") {
        return "Model answered directly via finish tool.";
    }

    return null;
}

function collectCitations(toolHistory: ToolInvocationRecord[]) {
    const citations = new Map<string, { label: string; jumpLink: string }>();
    for (const item of toolHistory) {
        if (item.tool !== T.retrieve_messages || !item.output.data || typeof item.output.data !== "object") {
            continue;
        }
        const rows =
            (item.output.data as { combinedResults?: Array<Record<string, unknown>> }).combinedResults ||
            [];
        for (const row of rows) {
            if (typeof row.jumpLink !== "string" || !row.jumpLink) {
                continue;
            }
            if (!citations.has(row.jumpLink)) {
                citations.set(row.jumpLink, {
                    label: row.channelName ? `#${String(row.channelName)}` : item.tool,
                    jumpLink: row.jumpLink,
                });
            }
        }
    }
    return [...citations.values()].slice(0, 6);
}

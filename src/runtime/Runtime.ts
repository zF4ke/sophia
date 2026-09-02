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
import { TOOL_DEFINITIONS, getToolDefinitions } from "@/runtime/toolSchemas";
import { formatDeferredInventory } from "@/tools/registry";
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
import type { DiscordToolEvidenceRole } from "@/shared/discordTools";
import { isMutatingTool, describeApproval, getToolEffect } from "@/tools/registry";
import { detectStallPromise, DoomLoopDetector, ProgressTracker } from "@/runtime/stallGuard";
import {
    buildForcedRetrieveArgs,
    buildRejectionMessage,
    createOrUpdate as updateCorpusTask,
    incompleteFallback as corpusIncompleteFallback,
    isIncomplete as corpusIsIncomplete,
    recordViolation as recordCorpusViolation,
    shouldForceContinuation as corpusShouldForceContinuation,
    shouldRejectFinish as corpusShouldRejectFinish,
    type CorpusTaskState,
} from "@/runtime/corpusTaskState";
import { compactMessages, shouldCompact } from "@/runtime/compaction";
import { compactInput, shouldCompactInput } from "@/runtime/inputCompaction";
import { countTokens } from "@/shared/tokenizer";
import type {
    DiscordToolResult,
    GroundedAnswerMode,
} from "@/shared/appTypes";

// ── Helpers: evidence extraction, identity formatting, retrieval session ──

/** Known tool-argument field names that must be valid Discord snowflake IDs. */
const SNOWFLAKE_FIELDS = new Set([
    "channel_id", "author_id", "member_id", "message_id",
    "role_id", "thread_id", "mentions", "before", "after",
    "authorId", "aroundMessageId",
]);

/** Known tool-argument array fields whose elements are snowflake IDs. */
const SNOWFLAKE_ARRAY_FIELDS = new Set([
    "message_ids", "channelIds", "excludedMessageIds", "role_ids",
]);

/**
 * Strip non-digit characters from snowflake ID fields in parsed tool arguments.
 * Mitigates LLM digit-hallucination (e.g. "73c78c50c77c04c582" → "7378507704582").
 * Only sanitizes values that look like corrupted snowflakes: mixed digit/non-digit
 * strings whose digit-only result is at least 15 chars (minimum Discord snowflake length).
 * Mutates `args` in place.
 */
function sanitizeSnowflakeArgs(args: ToolArguments): void {
    for (const key of Object.keys(args)) {
        const value = args[key];
        if (SNOWFLAKE_FIELDS.has(key) && typeof value === "string") {
            const cleaned = value.replace(/\D/g, "");
            if (cleaned !== value && cleaned.length >= 15) {
                args[key] = cleaned;
            }
        } else if (SNOWFLAKE_ARRAY_FIELDS.has(key) && Array.isArray(value)) {
            for (let i = 0; i < value.length; i++) {
                if (typeof value[i] === "string") {
                    const cleaned = (value[i] as string).replace(/\D/g, "");
                    if (cleaned !== value[i] && cleaned.length >= 15) {
                        value[i] = cleaned;
                    }
                }
            }
        }
    }
}

function looksLikeRawToolMarkup(text: string | null | undefined): boolean {
    if (!text) return false;
    return /<invoke\s+name="[^"]+"\s*>/i.test(text)
        || /<[a-z0-9_-]+:tool_call>/i.test(text)
        || /<\/[a-z0-9_-]+:tool_call>/i.test(text);
}

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

function buildCorrectionNote(correction?: string): string {
    return correction
        ? ` Admin correction: "${correction}". Adjust your approach based on this feedback.`
        : "";
}

function buildDeniedActionMessage(decidedBy: string, correction?: string): string {
    const correctionNote = buildCorrectionNote(correction);
    return `Action NOT executed. Denied by ${decidedBy}.${correctionNote} Do not tell the user this action was completed.`;
}

function buildStoppedActionMessage(decidedBy: string): string {
    return `Execution stopped by ${decidedBy}.`;
}

function createBlockedToolRecord(
    toolName: DiscordToolName,
    parsedArgs: ToolArguments,
    summary: string,
    learned: string
): ToolInvocationRecord {
    return {
        tool: toolName,
        arguments: parsedArgs,
        summary,
        learned,
        confidenceImproved: false,
        output: { tool: toolName, summary, data: null, errorMessage: learned },
        durationMs: 0,
        blocked: true,
    };
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
        .map((turn, index) => {
            const who = turn.requesterDisplayName?.trim() || "user";
            return `${index + 1}. ${who}: ${turn.question} | sophia: ${turn.answer}`;
        })
        .join("\n");
}

/**
 * Index freshness for the current channel: how old is the newest indexed
 * message, and is the full history already crawled? The agent compares this
 * against the current date to judge whether retrieval results are stale.
 */
async function buildIndexFreshness(channelId: string | null): Promise<string> {
    if (!channelId) return "Index status: unavailable (DM).";
    try {
        const [summary, crawlStates] = await Promise.all([
            DiscordMemoryService.getChannelSummaryAsync(channelId),
            DiscordMemoryService.getChannelCrawlStateAsync(channelId),
        ]);
        const crawl = crawlStates[0];
        const now = Date.now();
        if (!summary || summary.messageCount === 0) {
            return `- Channel <#${channelId}>: NOT indexed yet (0 messages). Use index_channel (mode "refresh" or "deep") before answering questions about its history.`;
        }
        const lastMsgAgeMs = summary.lastMessageTimestamp ? now - summary.lastMessageTimestamp : null;
        const lastMsgAge = lastMsgAgeMs == null
            ? "unknown"
            : lastMsgAgeMs < 3_600_000
                ? `${Math.round(lastMsgAgeMs / 60_000)} min`
                : lastMsgAgeMs < 86_400_000
                    ? `${Math.round(lastMsgAgeMs / 3_600_000)} h`
                    : `${Math.round(lastMsgAgeMs / 86_400_000)} days`;
        const exhausted = crawl?.exhausted ? "full history crawled" : "history backfill incomplete";
        return `- Channel <#${channelId}>: ${summary.messageCount} messages indexed, newest indexed message is ${lastMsgAge} old, ${exhausted}. Compare with today's date — if the user asks about recent activity and this looks stale, call index_channel first.`;
    } catch {
        return "Index status: unavailable.";
    }
}

function sanitizeAnswer(answer: string | null | undefined): string {
    const normalized = (answer || "").trim();
    const banned = new Set([
        "I couldn't ground that in Discord evidence.",
        "I don't have enough Discord evidence to answer that yet.",
    ]);
    return banned.has(normalized) ? "" : normalized;
}

function extractRequestedCorpusSize(question: string): number | null {
    const normalized = question.toLowerCase();
    const match = normalized.match(
        /\b(?:últimas|ultimas|last|based on|baseado em|com base em)?\s*(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(k)?\s*(?:mensagens|messages)\b/i,
    );
    if (!match) return null;
    const numeric = match[1].replace(/[.,]/g, "");
    const base = Number(numeric);
    if (!Number.isFinite(base) || base <= 0) return null;
    return match[2] ? base * 1000 : base;
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
// Align headroom with compaction trigger but keep a 3% margin so Tier-1 prunes
// before Tier-2 summarizes (inspired by opencode's PRUNE_PROTECT layer).
const CONTEXT_HEADROOM_RATIO = 0.85;

function isContextOverflow(promptTokens: number, contextWindow: number): boolean {
    return promptTokens >= contextWindow * CONTEXT_HEADROOM_RATIO;
}

function pruneOldToolOutputs(messages: ToolChatMessage[]): number {
    let pruned = 0;
    // Keep the system prompt (index 0), the user message (index 1),
    // and the last 8 messages (align with compaction PRESERVE_TAIL = 8).
    // Only prune older middle-block tool outputs; inspired by opencode's
    // prune-until-PROTECT approach but lightweight here.
    const protectedTail = 8;
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
        const requestedCorpusSize = extractRequestedCorpusSize(input.question);

        const traceEvents: RuntimeTraceEvent[] = [];
        const toolHistory: ToolInvocationRecord[] = [];
        const evidence: EvidenceItem[] = [];
        let confidence: GroundedAnswerMode = "insufficient";
        let stopReason: StopReason | null = null;
        let stallCorrectionsUsed = 0;
        let longTaskGrantedThisTurn = false;
        const evidenceRolesThisTurn = new Set<DiscordToolEvidenceRole>();
        const doomLoopDetector = new DoomLoopDetector();
        const progressTracker = new ProgressTracker();
        let corpusTask: CorpusTaskState | undefined;

        // Long-task caps come from configurable settings (runtime.longTask.*).
        const LONG_TASK_MAX_CALLS = config.runtime.longTask.maxToolCalls;
        const LONG_TASK_MAX_MS = config.runtime.longTask.maxLatencyBudgetMs;
        const LONG_TASK_EVIDENCE_FLOOR = config.runtime.longTask.evidenceSliceFloor;

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
            } else if (personality === "mixed") {
                try {
                    personalityOverride = PromptRegistry.load("system/personality_mixed_override");
                } catch {
                    personalityOverride = "";
                }
            }

            const contextFields = {
                recentTurns: summarizeRecentTurns(recentTurns),
                channelContext: formatChannelContext(channelContext),
                priorEvidence: priorEvidenceSummary,
                toolContext: formatRecentToolRuns(recentToolRuns),
            };

            const renderSystemPrompt = async (fields: typeof contextFields): Promise<string> =>
                PromptRegistry.render("runtime/agent_loop", {
                    guild_name: input.guild?.name || "DM",
                    guild_id: guildId || "none",
                    channel_name: channelId ? `<#${channelId}>` : "DM",
                    channel_id: channelId || "none",
                    requester_display_name: input.requesterDisplayName,
                    actor_id: actorId,
                    current_date: new Date().toISOString().slice(0, 10),
                    trigger: input.trigger,
                    recent_turns: fields.recentTurns,
                    channel_context: fields.channelContext,
                    prior_evidence: fields.priorEvidence,
                    tool_context: fields.toolContext,
                    reply_context: input.replyContext
                        ? `Replying to ${input.replyContext.authorDisplayName}: "${input.replyContext.content}"`
                        : "Not a reply.",
                    max_tool_calls: String(constraints.maxToolCalls),
                    personality_override: personalityOverride,
                    deferred_tools: formatDeferredInventory(),
                    index_freshness: await buildIndexFreshness(channelId),
                });

            let systemPrompt = await renderSystemPrompt(contextFields);

            // ── 3b. Tier-0 input compaction ──
            // Measure the assembled prompt with a tokenizer. If it's past the
            // input trigger fraction of the context window, summarise the bulky
            // context fields (recent turns, channel context, prior evidence,
            // tool context) into one narrative so tiny social replies don't
            // carry 30+ evidence items into the model.
            const preCompactionTokens = countTokens(systemPrompt) + countTokens(input.question);
            const willCompact = shouldCompactInput(
                preCompactionTokens,
                config.modelProfile.contextWindow,
            );
            if (willCompact) {
                await input.debugSession?.setToolRunning?.("compact_context", [
                    `~${preCompactionTokens} tokens over threshold`,
                ]);
            }
            const compactionOutcome = await compactInput({
                assembledPrompt: systemPrompt,
                question: input.question,
                contextWindow: config.modelProfile.contextWindow,
                fields: contextFields,
                traceEvents,
            });
            if (compactionOutcome.compacted && compactionOutcome.fields) {
                systemPrompt = await renderSystemPrompt(compactionOutcome.fields);
                trace(
                    "compaction_tier0_applied",
                    `Input compacted ${compactionOutcome.promptTokensBefore} → ~${compactionOutcome.promptTokensAfter} tokens.`,
                );
                await input.debugSession?.setToolResult?.(
                    "compact_context",
                    `${compactionOutcome.promptTokensBefore} → ~${compactionOutcome.promptTokensAfter} tokens`,
                );
            } else if (willCompact) {
                await input.debugSession?.setToolResult?.(
                    "compact_context",
                    "skipped (no compactable context or summariser unavailable)",
                );
            }

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
            let malformedToolCallCorrections = 0;
            // Dynamic tool discovery — deferred tools are loaded via tool_search (opencode/codex pattern).
            const discoveredTools = new Set<string>();

            /**
             * Runtime-driven forced continuation: when the unified corpus guard
             * has rejected the model twice, stop asking it to retry and invoke
             * retrieve_messages ourselves. Updates corpusTask from the real
             * result and pushes a synthetic assistant+tool message pair into
             * the conversation so the model sees the new page on the next turn.
             */
            const forceCorpusContinuation = async (): Promise<boolean> => {
                if (!corpusShouldForceContinuation(corpusTask)) return false;
                // Local budget gate — never spend a forced call past the
                // tool-count or latency cap. If we're out of budget, fall
                // through to the deterministic incomplete-progress fallback.
                if (totalToolCalls >= constraints.maxToolCalls) {
                    trace("corpus_force_continuation_skipped", `tool budget exhausted (${totalToolCalls}/${constraints.maxToolCalls}).`);
                    return false;
                }
                const elapsedMs = getActiveElapsedMs();
                if (elapsedMs >= constraints.maxLatencyBudgetMs) {
                    trace("corpus_force_continuation_skipped", `latency budget exhausted (${elapsedMs}ms/${constraints.maxLatencyBudgetMs}ms).`);
                    return false;
                }
                const capability = CapabilityRegistry.get(T.retrieve_messages);
                if (!capability) return false;
                const forcedArgs = buildForcedRetrieveArgs(corpusTask);
                const syntheticCallId = `forced_${randomUUID()}`;
                trace(
                    "corpus_force_continuation",
                    `Runtime forcing retrieve_messages: collected=${corpusTask.collected}/${corpusTask.requested}.`,
                );
                const toolStartedAt = Date.now();
                try {
                    const output = await capability.run(
                        {
                            guild: input.guild || null,
                            question: input.question,
                            currentChannelId: channelId,
                            requestId,
                            threadId,
                            onProgress: async (tn, summary) => {
                                trace("tool_progress", `${tn}: ${summary}`);
                                await input.debugSession?.setToolProgress?.(tn, summary);
                            },
                        },
                        forcedArgs,
                    );
                    const toolDurationMs = Date.now() - toolStartedAt;
                    const evidenceItems = extractEvidence(output);
                    const learned = evidenceItems.map((i) => i.content).join(" | ") || output.summary;
                    const retrieval = getRetrievalSummary(output);
                    const record: ToolInvocationRecord = {
                        tool: T.retrieve_messages,
                        arguments: forcedArgs,
                        summary: output.summary,
                        learned,
                        confidenceImproved: evidenceItems.some((i) => i.strength !== "weak"),
                        output,
                        durationMs: toolDurationMs,
                        retrievalSummary: retrieval,
                    };
                    totalToolCalls += 1;
                    toolHistory.push(record);
                    evidence.push(...evidenceItems);
                    for (const item of evidenceItems) evidenceRolesThisTurn.add(item.evidenceRole);

                    trace("tool_result", `${T.retrieve_messages} (forced): ${output.summary}`);

                    // Persist the forced tool run just like a normal one so
                    // the operational store, traces, and debug surfaces stay
                    // in sync with what actually happened.
                    await DiscordMemoryService.recordToolRun(
                        requestId,
                        guildId,
                        channelId,
                        actorId,
                        input.question,
                        T.retrieve_messages,
                        JSON.stringify(forcedArgs),
                        output.summary,
                        learned,
                        JSON.stringify(output),
                        record.confidenceImproved,
                        toolDurationMs,
                    );
                    await input.debugSession?.setToolResult(T.retrieve_messages, output.summary);
                    if (retrieval) {
                        await input.debugSession?.setRetrievalSummary?.(retrieval);
                    }

                    // Update state from the real retrieval result and clear the
                    // unified violation counter so the model gets a fresh chance.
                    corpusTask = updateCorpusTask(
                        corpusTask,
                        retrieval,
                        forcedArgs,
                        settings.runtime.retrievalHistoryLimit,
                        requestedCorpusSize,
                    );
                    if (corpusTask) corpusTask.guardViolations = 0;
                    trace(
                        "corpus_task",
                        `(forced) collected=${corpusTask?.collected}/${corpusTask?.requested} continuationAvailable=${corpusTask?.continuationAvailable} historyExhausted=${corpusTask?.historyExhausted}`,
                    );

                    messages.push({
                        role: "assistant",
                        content: null,
                        tool_calls: [{
                            id: syntheticCallId,
                            type: "function",
                            function: {
                                name: T.retrieve_messages,
                                arguments: JSON.stringify(forcedArgs),
                            },
                        }],
                    });
                    const resultPayload = output.errorMessage
                        ? JSON.stringify({ error: output.errorMessage })
                        : JSON.stringify({ summary: output.summary, data: output.data });
                    messages.push({
                        role: "tool",
                        tool_call_id: syntheticCallId,
                        content: truncateToolResult(resultPayload),
                    });
                    return true;
                } catch (err) {
                    const message = (err as { message?: unknown })?.message;
                    trace("corpus_force_continuation_error", typeof message === "string" ? message : "unknown");
                    return false;
                }
            };

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

                // ── Plan/notes header: inject transient system message so plan
                //    survives compaction and is re-seen every iteration. ──
                let planHeaderIndex: number | null = null;
                try {
                    const planBody = await DiscordMemoryService.getRequestPlan(requestId);
                    if (planBody) {
                        const notesCount = await DiscordMemoryService.countRequestNotes({
                            requestId,
                            kind: "note",
                        });
                        const header =
                            `Current plan:\n${planBody}\n` +
                            `Notes so far: ${notesCount} (call note_list to read them).`;
                        messages.push({ role: "system", content: header });
                        planHeaderIndex = messages.length - 1;
                    }
                } catch { /* non-fatal */ }

                const dynamicTools = getToolDefinitions(discoveredTools);
                const result = await ModelGateway.generateWithTools(messages, {
                    tools: dynamicTools,
                    traceContext: {
                        traceLabel: `agent_loop_iter_${iteration}`,
                        questionPreview: input.question,
                        traceEvents: [...traceEvents],
                    },
                });

                // Remove the transient plan-header so it's re-synthesized next turn.
                if (planHeaderIndex !== null) {
                    messages.splice(planHeaderIndex, 1);
                }

                // Track token usage
                if (result.usage) {
                    cumulativePromptTokens += result.usage.promptTokens;
                    cumulativeCompletionTokens += result.usage.completionTokens;
                    const ctxWindow = config.modelProfile.contextWindow;
                    const pctOfWindow = ctxWindow > 0
                        ? ((result.usage.promptTokens / ctxWindow) * 100).toFixed(1)
                        : "?";
                    trace(
                        `iter_${iteration}_tokens`,
                        `prompt=${result.usage.promptTokens} (${pctOfWindow}% of ${ctxWindow}) | completion=${result.usage.completionTokens} | cumulative=${cumulativePromptTokens}/${cumulativeCompletionTokens}`,
                    );
                }

                // Context overflow guard — prune old tool outputs if approaching limit
                if (result.usage && isContextOverflow(result.usage.promptTokens, config.modelProfile.contextWindow)) {
                    const pruned = pruneOldToolOutputs(messages);
                    if (pruned > 0) {
                        trace("context_prune", `Pruned ${pruned} old tool outputs (prompt tokens: ${result.usage.promptTokens}/${config.modelProfile.contextWindow}).`);
                    }

                    // Tier-2 compaction: if Tier-1 wasn't enough, summarise the middle block.
                    if (shouldCompact(result.usage.promptTokens, config.modelProfile.contextWindow)) {
                        const compactionResult = await compactMessages({
                            messages,
                            promptTokens: result.usage.promptTokens,
                            contextWindow: config.modelProfile.contextWindow,
                            traceEvents,
                        });
                        if (compactionResult.compacted) {
                            trace("compaction_tier2", `Compacted ${compactionResult.removedCount} messages (~${compactionResult.summaryTokenEstimate} token summary).`);
                        }
                    }
                }

                // No tool calls → model produced a text response (shouldn't happen with finish tool, but handle it)
                if (result.toolCalls.length === 0) {
                    if (result.malformedToolCallText || looksLikeRawToolMarkup(result.content)) {
                        malformedToolCallCorrections += 1;
                        trace("malformed_tool_call", "Model emitted raw invoke markup instead of structured tool_calls.");
                        messages.push({
                            role: "system",
                            content:
                                "Your previous response emitted raw tool markup instead of a structured function call. " +
                                "Retry the same next step using proper tool_calls only. Do not output <invoke>, XML, or pseudo-tool syntax.",
                        });
                        if (malformedToolCallCorrections >= 2) {
                            stopReason = "no_useful_next_step";
                            trace("stop", "Model repeated malformed raw tool markup twice.");
                            break;
                        }
                        continue;
                    }
                    if (result.content && corpusShouldRejectFinish(corpusTask)) {
                        recordCorpusViolation(corpusTask);
                        trace(
                            "corpus_guard",
                            `Rejected raw-text answer: collected=${corpusTask.collected}/${corpusTask.requested} (violation ${corpusTask.guardViolations}).`,
                        );
                        messages.push({
                            role: "assistant",
                            content: result.content,
                        });
                        messages.push({
                            role: "system",
                            content: buildRejectionMessage(corpusTask),
                        });
                        await forceCorpusContinuation();
                        continue;
                    }
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
                    // Notify progress if available
                    if (input.progressNotifier) {
                        await input.progressNotifier(`Running ${toolName}…`).catch(() => {});
                    }
                    const toolStartedAt = Date.now();
                    try {
                        const output = await capability.run(
                            {
                                guild: input.guild || null,
                                question: input.question,
                                currentChannelId: channelId,
                                requestId,
                                threadId,
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
                        for (const item of evidenceItems) {
                            evidenceRolesThisTurn.add(item.evidenceRole);
                        }

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
                        // Dynamic discovery: if tool_search returned results, expose them next iteration.
                        if (toolName === T.tool_search && output.data && typeof output.data === "object") {
                            try {
                                const d = output.data as { results?: { name: string }[] };
                                for (const r of d.results ?? []) {
                                    if (r?.name) discoveredTools.add(r.name);
                                }
                                if (d.results?.length) trace("tool_discovery", `Discovered: ${d.results.map((r) => r.name).join(", ")}`);
                            } catch { /* best-effort */ }
                        }
                        if (toolName === T.retrieve_messages && retrieval) {
                            const prevViolations = corpusTask?.guardViolations ?? 0;
                            corpusTask = updateCorpusTask(
                                corpusTask,
                                retrieval,
                                parsedArgs,
                                settings.runtime.retrievalHistoryLimit,
                                requestedCorpusSize,
                            );
                            if (corpusTask) {
                                // Preserve the unified violation counter across updates.
                                corpusTask.guardViolations = prevViolations;
                                trace(
                                    "corpus_task",
                                    `collected=${corpusTask.collected}/${corpusTask.requested} continuationAvailable=${corpusTask.continuationAvailable} historyExhausted=${corpusTask.historyExhausted}`,
                                );
                            }
                        }
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
                    } catch (error) {
                        const toolDurationMs = Date.now() - toolStartedAt;
                        totalToolCalls += 1;

                        const err = error as {
                            message?: unknown;
                            code?: unknown;
                            status?: unknown;
                            rawError?: unknown;
                        };
                        const messageText = typeof err.message === "string"
                            ? err.message
                            : "Unknown tool error.";
                        const codeText = err.code == null ? "" : ` (code: ${String(err.code)})`;
                        const statusText = err.status == null ? "" : ` (status: ${String(err.status)})`;
                        const rawText = err.rawError == null
                            ? ""
                            : ` Raw: ${JSON.stringify(err.rawError)}`;
                        const toolErrorText = `Tool ${toolName} failed: ${messageText}${codeText}${statusText}.${rawText}`;

                        trace("tool_error", toolErrorText);

                        const failedOutput = {
                            tool: toolName,
                            summary: `Failed to execute ${toolName}.`,
                            data: null,
                            errorMessage: toolErrorText,
                        };

                        messages.push({
                            role: "tool",
                            tool_call_id: tc.id,
                            content: truncateToolResult(JSON.stringify({ error: toolErrorText })),
                        });

                        toolHistory.push({
                            tool: toolName,
                            arguments: parsedArgs,
                            summary: failedOutput.summary,
                            learned: toolErrorText,
                            confidenceImproved: false,
                            output: failedOutput,
                            durationMs: toolDurationMs,
                            blocked: true,
                        });

                        await input.debugSession?.setToolResult(toolName, failedOutput.summary);
                    }
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

                    // ── Sanitize snowflake ID fields (mitigates LLM digit-hallucination) ──
                    sanitizeSnowflakeArgs(parsedArgs);

                    // ── Handle "start_long_task" tool (intercepted — never dispatched) ──
                    if (toolName === "start_long_task") {
                        if (longTaskGrantedThisTurn) {
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ ok: true, note: "Budget already raised this turn." }),
                            });
                            trace("long_task", "Idempotent — budget already raised this turn.");
                        } else {
                            // Estimates are ignored — the model cannot reliably predict work size.
                            // We always raise budgets to the operator-configured long-task caps.
                            const newMaxCalls = Math.max(constraints.maxToolCalls, LONG_TASK_MAX_CALLS);
                            const newMaxMs = Math.max(constraints.maxLatencyBudgetMs, LONG_TASK_MAX_MS);
                            constraints.maxToolCalls = newMaxCalls;
                            constraints.maxLatencyBudgetMs = newMaxMs;
                            // Raise per-turn evidence slice so large retrievals aren't dropped.
                            constraints.maxEvidenceSlice = Math.max(constraints.maxEvidenceSlice, LONG_TASK_EVIDENCE_FLOOR);
                            longTaskGrantedThisTurn = true;
                            if (input.progressNotifier) {
                                await input.progressNotifier("Extending runtime budget for a long task…").catch(() => {});
                            }
                            trace("long_task", `Budget raised: calls=${newMaxCalls}, latency=${newMaxMs}ms. reason=${String(parsedArgs.reason ?? "none")}`);
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ ok: true, maxToolCalls: newMaxCalls, maxLatencyBudgetMs: newMaxMs }),
                            });
                        }
                        continue;
                    }

                    // ── Handle "finish" tool ──
                    if (toolName === "finish") {
                        answer = (parsedArgs.answer as string) || result.content || "";

                        // ── Corpus task guard: reject finish while under explicit target ──
                        if (corpusShouldRejectFinish(corpusTask)) {
                            recordCorpusViolation(corpusTask);
                            const rejection = buildRejectionMessage(corpusTask);
                            trace(
                                "corpus_guard",
                                `Rejected finish: collected=${corpusTask.collected}/${corpusTask.requested} (violation ${corpusTask.guardViolations}).`,
                            );
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ error: rejection }),
                            });
                            answer = "";
                            // After the threshold, the runtime stops asking the model
                            // nicely and executes retrieve_messages itself.
                            await forceCorpusContinuation();
                            continue;
                        }

                        // ── Stall guard: detect empty-promise answers ──
                        if (stallCorrectionsUsed < 2) {
                            const stallResult = await detectStallPromise({
                                answer,
                                toolHistoryThisTurn: toolHistory,
                                evidenceRoles: evidenceRolesThisTurn,
                                getToolEffect,
                                longTaskGranted: longTaskGrantedThisTurn,
                            });
                            if (stallResult.stalled) {
                                stallCorrectionsUsed += 1;
                                const correction = stallCorrectionsUsed === 1
                                    ? "Your answer promises action without having done anything. Call the appropriate tools now — use retrieve_messages or search_messages to gather evidence before finishing."
                                    : "You are STILL finishing without evidence. Do NOT finish yet. Call retrieve_messages or search_messages RIGHT NOW.";
                                trace("stall_guard", `Stalled on "${stallResult.matchedPhrase}" (correction ${stallCorrectionsUsed}/2).`);
                                messages.push({
                                    role: "tool",
                                    tool_call_id: tc.id,
                                    content: JSON.stringify({ error: correction }),
                                });
                                answer = "";
                                continue;
                            }
                        }

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
                            const stopMsg = buildStoppedActionMessage(approvalResult.decidedBy);
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ error: stopMsg }),
                            });
                            const stoppedRecord = createBlockedToolRecord(
                                toolName,
                                parsedArgs,
                                `Execution stopped by ${approvalResult.decidedBy}.`,
                                stopMsg
                            );
                            toolHistory.push(stoppedRecord);
                            trace("stop", `${toolName}: execution stopped by ${approvalResult.decidedBy}`);
                            stopReason = "execution_stopped_by_admin";
                            loopDone = true;
                            await input.activityIndicator?.startThinking();
                            break;
                        }
                        if (!approvalResult.approved) {
                            const correctionNote = buildCorrectionNote(approvalResult.correction);
                            const denyMsg = buildDeniedActionMessage(
                                approvalResult.decidedBy,
                                approvalResult.correction
                            );
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ error: denyMsg }),
                            });
                            const deniedRecord = createBlockedToolRecord(
                                toolName,
                                parsedArgs,
                                `Denied by ${approvalResult.decidedBy}.${correctionNote}`,
                                denyMsg
                            );
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
                    {
                        // Doom-loop guard: detect repeated identical calls.
                        const argsHash = JSON.stringify(parsedArgs);
                        const doomResult = doomLoopDetector.recordCall(toolName, argsHash);
                        if (doomResult.action === "force_finish") {
                            trace("doom_loop_force", doomResult.message ?? "Force-finishing due to doom loop.");
                            messages.push({
                                role: "tool",
                                tool_call_id: tc.id,
                                content: JSON.stringify({ error: doomResult.message }),
                            });
                            stopReason = "confidence_plateau";
                            loopDone = true;
                            break;
                        }
                        if (doomResult.action === "nudge") {
                            trace("doom_loop_nudge", doomResult.message ?? "Doom loop nudge.");
                            // Still execute, but inject a nudge after.
                        }

                        const evidenceBefore = evidence.length;
                        await executeToolCall(tc, toolName, parsedArgs);
                        const producedEvidence = evidence.length > evidenceBefore;

                        // Inject doom-loop nudge after execution.
                        if (doomResult.action === "nudge") {
                            messages.push({
                                role: "system",
                                content: doomResult.message!,
                            });
                        }

                        // Progress-required check (long-task only).
                        if (longTaskGrantedThisTurn) {
                            const progressResult = progressTracker.recordCall(toolName, producedEvidence);
                            if (progressResult.stalled) {
                                trace("progress_stall", progressResult.message ?? "No progress detected.");
                                messages.push({
                                    role: "system",
                                    content: progressResult.message!,
                                });
                            }
                        }
                    }
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
                            const stopMsg = buildStoppedActionMessage(approvalResult.decidedBy);
                            messages.push({ role: "tool", tool_call_id: item.tc.id, content: JSON.stringify({ error: stopMsg }) });
                            const stoppedRecord = createBlockedToolRecord(
                                item.toolName,
                                item.parsedArgs,
                                `Execution stopped by ${approvalResult.decidedBy}.`,
                                stopMsg
                            );
                            toolHistory.push(stoppedRecord);
                            trace("stop", `${item.toolName}: execution stopped by ${approvalResult.decidedBy}`);
                            stopReason = "execution_stopped_by_admin";
                            loopDone = true;
                            await input.activityIndicator?.startThinking();
                        } else if (!approvalResult.approved) {
                            const correctionNote = buildCorrectionNote(approvalResult.correction);
                            const denyMsg = buildDeniedActionMessage(
                                approvalResult.decidedBy,
                                approvalResult.correction
                            );
                            messages.push({ role: "tool", tool_call_id: item.tc.id, content: JSON.stringify({ error: denyMsg }) });
                            const deniedRecord = createBlockedToolRecord(
                                item.toolName,
                                item.parsedArgs,
                                `Denied by ${approvalResult.decidedBy}.${correctionNote}`,
                                denyMsg
                            );
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
                                const stopMsg = buildStoppedActionMessage(batchResult.decidedBy);
                                messages.push({
                                    role: "tool",
                                    tool_call_id: item.tc.id,
                                    content: JSON.stringify({ error: stopMsg }),
                                });
                                const stoppedRecord = createBlockedToolRecord(
                                    item.toolName,
                                    item.parsedArgs,
                                    stopMsg,
                                    stopMsg
                                );
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
                                    const denyMsg = buildDeniedActionMessage(
                                        batchResult.decidedBy,
                                        batchResult.correction
                                    );
                                    messages.push({
                                        role: "tool",
                                        tool_call_id: item.tc.id,
                                        content: JSON.stringify({ error: denyMsg }),
                                    });
                                    const deniedRecord = createBlockedToolRecord(
                                        item.toolName,
                                        item.parsedArgs,
                                        `Denied by ${batchResult.decidedBy}.${batchCorrectionNote}`,
                                        denyMsg
                                    );
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

                // Corpus task incomplete → do NOT synthesize a "finished" answer
                // from model-authored notes (which may contradict runtime facts).
                // Return a deterministic incomplete-progress string instead.
                if (corpusIsIncomplete(corpusTask)) {
                    answer = corpusIncompleteFallback(corpusTask);
                    confidence = "insufficient";
                    trace(
                        "corpus_incomplete_fallback",
                        `collected=${corpusTask.collected}/${corpusTask.requested} — returning deterministic incomplete-progress string.`,
                    );
                }

                // For long tasks, try to build answer from plan + notes.
                if (!answer && longTaskGrantedThisTurn) {
                    try {
                        const planBody = await DiscordMemoryService.getRequestPlan(requestId);
                        const notes = await DiscordMemoryService.listRequestNotes({
                            requestId,
                            threadId,
                            includeThreadHistory: false,
                            kind: "note",
                        });
                        if (planBody || notes.length > 0) {
                            const notesText = notes.slice(-20).map(n => `- ${n.body}`).join("\n");
                            const contextBlock = [
                                planBody ? `Plan:\n${planBody}` : null,
                                notesText ? `Notes:\n${notesText}` : null,
                            ].filter(Boolean).join("\n\n");
                            const synthMessages = [
                                { role: "system" as const, content: systemPrompt },
                                { role: "user" as const, content: input.question },
                                { role: "assistant" as const, content: `Here is what I found during my research:\n${contextBlock}` },
                                { role: "user" as const, content: "Based on that research, answer the original question as concisely as possible. Cite jumpLinks where available. If you truly could not find it, say so in one sentence." },
                            ];
                            const synthResult = await ModelGateway.generateText(synthMessages, {
                                traceContext: { traceLabel: "long_task_synthesis", questionPreview: input.question, traceEvents: [...traceEvents] },
                            });
                            const cleaned = sanitizeAnswer(synthResult);
                            if (cleaned) {
                                answer = cleaned;
                                confidence = "best_effort";
                                trace("long_task_synthesis", `Synthesized answer from plan + ${notes.length} notes.`);
                            }
                        }
                    } catch { /* non-fatal */ }
                }

                if (!answer) {
                    confidence = toolHistory.length > 0
                        ? answerConfidenceForInsufficient({ evidence })
                        : "insufficient";
                    answer = toolHistory.length > 0
                        ? await synthesizeAnswer(systemPrompt, input.question, toolHistory, evidence, traceEvents)
                        : "";
                }
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
                requesterDisplayName: input.requesterDisplayName || null,
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

            // ── 7b. Notes snapshot for debug ──
            if (input.debugSession?.setNotesSnapshot) {
                const planBody = await DiscordMemoryService.getRequestPlan(requestId);
                const noteRecords = await DiscordMemoryService.listRequestNotes({ requestId, kind: "note" });
                if (noteRecords.length > 0 || planBody) {
                    const snapshot = noteRecords.map((n) => ({
                        seq: n.seq,
                        label: n.label,
                        bodyPreview: n.body.slice(0, 200),
                        wordCount: n.body.split(/\s+/).filter(Boolean).length,
                    }));
                    const totalWords = snapshot.reduce((sum, n) => sum + n.wordCount, 0);
                    trace("notes_snapshot", `${snapshot.length} note(s), ${totalWords} words`);
                    await input.debugSession.setNotesSnapshot(snapshot, planBody);
                }
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
            // Surface a one-line failure to the user — an empty answer reads as
            // "Sophia ignored me", which is worse than admitting the fault.
            const userHint = error instanceof Error && error.message
                ? error.message.slice(0, 120)
                : "erro interno";
            return {
                requestId,
                threadId,
                answer: `Não consegui processar isso agora (${userHint}). Tenta de novo — se persistir, o problema é meu, não teu.`,
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

    if (stopReason === "stalled_promise") {
        return findLast((event) => event.label === "stall_guard");
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

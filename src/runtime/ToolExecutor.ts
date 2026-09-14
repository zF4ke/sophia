import { ExecutionStopped } from "./ExecutionControl";
import { assertReadableChannels } from "@/security/SourceAccess";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { randomUUID } from "node:crypto";
import { taskStore } from "./tasks/TaskStore";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import type {
    EvidenceItem,
    RetrievalSummary,
    ToolArguments,
    ToolInvocationRecord,
} from "@/runtime/contracts";
import { getToolStrategy, isMutatingTool, getPublicationTarget } from "@/tools/registry";
import { assertPublicationAudience } from "@/security/DerivedSources";
import { KeyedLock } from "@/shared/KeyedLock";
import { SettingsService } from "@/app/SettingsService";
import type { CapabilityContext } from "@/tools/types";
import type { DiscordToolName } from "@/shared/discordTools";
import type { DiscordToolResult } from "@/shared/appTypes";

export interface ToolExecutionResult {
    images?: DiscordToolResult["images"];
    uncertainAction?: boolean;
    record: ToolInvocationRecord;
    evidence: EvidenceItem[];
    retrievalSummary: RetrievalSummary | null;
    resultPayload: string;
}

export interface ToolExecutionOptions {
    invocationId?: string;
    /** Supplied only by the caller after approval of this exact invocation. */
    approved?: boolean;
    /** Reject an unstarted call if its model decision predates a correction. */
    steeringRevision?: number;
    /** Read-only capabilities are bounded so one stuck dependency cannot hang a turn. */
    timeoutMs?: number;
}

function errorDetails(error: unknown): string {
    if (error == null || typeof error !== "object") return String(error);
    const value = error as {
        message?: unknown;
        code?: unknown;
        status?: unknown;
        rawError?: unknown;
    };
    const message = typeof value.message === "string" ? value.message : "Unknown tool error.";
    const code = value.code == null ? "" : ` (code: ${String(value.code)})`;
    const status = value.status == null ? "" : ` (status: ${String(value.status)})`;
    let raw = "";
    try { if (value.rawError != null) raw = ` Raw: ${JSON.stringify(value.rawError)}`; } catch { /* Error reporting must survive circular provider data. */ }
    return `${message}${code}${status}.${raw}`;
}

async function withTimeout<T>(operation: Promise<T>, timeoutMs: number): Promise<T> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
        return await Promise.race([
            operation,
            new Promise<never>((_, reject) => {
                timer = setTimeout(
                    () => reject(new Error(`Timed out after ${timeoutMs}ms`)),
                    timeoutMs,
                );
                timer.unref?.();
            }),
        ]);
    } finally {
        if (timer) clearTimeout(timer);
    }
}

/**
 * The sole module that invokes a capability and converts its outcome into the
 * runtime's durable record shape. Approval remains a caller responsibility,
 * so no alternate caller can accidentally claim that a mutation was approved.
 */
export class ToolExecutor {
    private static readonly mutations = new KeyedLock();
    public static async execute(
        toolName: DiscordToolName,
        args: ToolArguments,
        context: CapabilityContext,
        options: ToolExecutionOptions = {},
    ): Promise<ToolExecutionResult> {
        const execute = () => this.executeUnlocked(toolName, args, context, options);
        // Server mutations share resources through roles, categories and channels.
        // Keep one ordered writer per guild; independent guilds and reads proceed.
        return isMutatingTool(toolName)
            ? this.mutations.run(context.guild?.id ?? `dm:${context.currentChannelId ?? context.actorId ?? "unknown"}`, execute)
            : execute();
    }

    private static async executeUnlocked(
        toolName: DiscordToolName,
        args: ToolArguments,
        context: CapabilityContext,
        options: ToolExecutionOptions = {},
    ): Promise<ToolExecutionResult> {
        await context.execution?.flushSteering();
        context.execution?.beforeTool();
        const startedAt = Date.now();
        let actionId: string | undefined;
        let dispatched = false;
        let actionRecorded = false;
        let returnedOutput: DiscordToolResult | undefined;
        try {
            if (context.execution?.taskResumeSelected) throw new Error("A prior task has been selected. Call finish to continue it before doing more work.");
            const capability = CapabilityRegistry.get(toolName);
            const decision = context.authorize ? await context.authorize(capability.sideEffectLevel, toolName) : null;
            if (decision === "deny") {
                throw new Error("Access revoked or action outside the requester grant.");
            }
            if (decision === "ask" && !options.approved) throw new Error("This action requires approval before execution.");
            const validation = capability.inputSchema.safeParse(args);
            if (!validation.success) {
                throw new Error(`Invalid arguments: ${validation.error.issues
                    .map((issue) => `${issue.path.join(".") || "input"}: ${issue.message}`)
                    .join("; ")}`);
            }
            const validatedArgs = validation.data as ToolArguments;
            const assertProtectedTarget = () => {
                if (capability.sideEffectLevel === "destructive" && typeof validatedArgs.channel_id === "string" && SettingsService.load().protectedChannelIds.includes(validatedArgs.channel_id)) throw new Error("This channel is protected from destructive or access-sensitive changes.");
            };
            assertProtectedTarget();
            if (options.steeringRevision !== undefined && context.execution?.steeringRevision !== options.steeringRevision) {
                throw new Error("Skipped: a user correction superseded this unstarted action. Reconsider it using the new instruction.");
            }
            if (capability.sideEffectLevel !== "none" && context.taskId) {
                if (!context.actorId) throw new Error("Action owner is missing.");
                actionId = await taskStore.beginAction({ taskId: context.taskId, actorId: context.actorId,
                    channelId: context.currentChannelId ?? "", guildId: context.guild?.id ?? null,
                    invocationId: options.invocationId ?? randomUUID(), tool: toolName, arguments: validatedArgs });
                // Recording can yield to steering or revocation. Recheck at dispatch.
                const latest = context.authorize ? await context.authorize(capability.sideEffectLevel, toolName) : null;
                if (latest === "deny" || (latest === "ask" && !options.approved)) throw new Error("Action authority changed before dispatch.");
                if (context.execution?.signal.aborted) throw new ExecutionStopped(context.execution.sourceInvalidated ? "source_changed" : "cancelled");
                if (options.steeringRevision !== undefined && context.execution?.steeringRevision !== options.steeringRevision) {
                    throw new Error("Skipped: requester correction arrived before dispatch.");
                }
            }
            const publicationTarget = await getPublicationTarget(toolName, context, validatedArgs);
            if (publicationTarget !== undefined) await assertPublicationAudience(context, publicationTarget);
            if (context.execution?.signal.aborted) throw new ExecutionStopped(context.execution.sourceInvalidated ? "source_changed" : "cancelled");
            if (options.steeringRevision !== undefined && context.execution?.steeringRevision !== options.steeringRevision) throw new Error("Skipped: requester correction arrived before dispatch.");
            const dispatchDecision = context.authorize ? await context.authorize(capability.sideEffectLevel, toolName) : null;
            if (dispatchDecision === "deny" || (dispatchDecision === "ask" && !options.approved)) throw new Error("Action authority changed before dispatch.");
            assertProtectedTarget();
            if (context.execution?.signal.aborted) throw new ExecutionStopped(context.execution.sourceInvalidated ? "source_changed" : "cancelled");
            if (options.steeringRevision !== undefined && context.execution?.steeringRevision !== options.steeringRevision) throw new Error("Skipped: requester correction arrived before dispatch.");
            dispatched = true;
            const run = capability.run(context, validatedArgs);
            const timeoutMs = Math.max(1, options.timeoutMs ?? 0);
            const rawOutput = capability.sideEffectLevel === "none" && options.timeoutMs
                ? await withTimeout(run, timeoutMs)
                : await run;
            returnedOutput = rawOutput;
            if (rawOutput.errorMessage) throw new Error(rawOutput.errorMessage);
            const outputValidation = capability.outputSchema.safeParse(rawOutput.data);
            if (!outputValidation.success) {
                throw new Error(`Invalid capability data: ${outputValidation.error.message}`);
            }
            const { images, ...storedOutput } = rawOutput;
            const output: DiscordToolResult = {
                ...storedOutput,
                data: outputValidation.data,
            };
            if (actionId) {
                await taskStore.settleAction(actionId, context.actorId!, "succeeded", output);
                actionRecorded = true;
            }
            const durationMs = Date.now() - startedAt;
            const strategy = getToolStrategy(toolName);
            const evidence = isMutatingTool(toolName) ? [] : strategy.extractEvidence(output);
            const messageEvidence = evidence.filter(item => item.messageId);
            context.execution?.watchSources(messageEvidence.flatMap(item => item.messageId ? [item.messageId] : []));
            if (await taskStore.hasDeletedSources(messageEvidence.flatMap(item => item.messageId ? [item.messageId] : []))) throw new Error("A retrieved source was deleted before delivery.");
            if (context.actorId && context.guild && messageEvidence.length) {
                await assertReadableChannels(context.guild, context.actorId, [...new Set(messageEvidence.flatMap(item => item.channelId ? [item.channelId] : []))], { client: context.client, privateResponse: context.privateResponse, destinationChannelId: context.currentChannelId });
                for (const item of messageEvidence) {
                    if (item.jumpLink && await knowledgeStore.isSourceInvalid(item.jumpLink)) throw new Error("A retrieved source was deleted before delivery. Retrieve current evidence again.");
                }
            }
            if (context.taskId && context.actorId && context.requestId && messageEvidence.length) await taskStore.recordEvidenceSources({ taskId: context.taskId,
                actorId: context.actorId, channelId: context.currentChannelId ?? "", guildId: context.guild?.id ?? null, requestId: context.requestId },
                messageEvidence.map(item => ({ messageId: item.messageId!, channelId: item.channelId, sourceUrl: item.jumpLink })));
            const retrievalSummary = isMutatingTool(toolName)
                ? null
                : strategy.extractRetrievalSummary?.(output) ?? null;
            const learned = evidence.map((item) => item.content).join(" | ") || output.summary;
            return {
                images,
                record: {
                    tool: toolName,
                    arguments: validatedArgs,
                    summary: output.summary,
                    learned,
                    confidenceImproved: evidence.some((item) => item.strength !== "weak"),
                    output,
                    durationMs,
                    retrievalSummary,
                },
                evidence,
                retrievalSummary,
                resultPayload: output.errorMessage
                    ? JSON.stringify({ error: output.errorMessage })
                    : JSON.stringify({ summary: output.summary, data: output.data }),
            };
        } catch (error) {
            const uncertainAction = Boolean(actionId && dispatched && !actionRecorded);
            if (actionId && !actionRecorded) {
                await taskStore.settleAction(actionId, context.actorId!, dispatched ? "unknown" : "skipped", returnedOutput ?? null,
                    error instanceof Error ? error.message : String(error)).catch(() => undefined);
            }
            if (error instanceof ExecutionStopped) throw error;
            const durationMs = Date.now() - startedAt;
            const detail = uncertainAction
                ? `Outcome of ${toolName} is unknown. Do not retry this action without verifying its effect. Receipt: ${actionId}. ${errorDetails(error)}`
                : `Tool ${toolName} failed: ${errorDetails(error)}`;
            const output = {
                tool: toolName,
                summary: uncertainAction ? `Outcome of ${toolName} is unknown (receipt ${actionId}).` : `Failed to execute ${toolName}.`,
                data: null,
                errorMessage: detail,
            };
            return {
                uncertainAction,
                record: {
                    tool: toolName,
                    arguments: args,
                    summary: output.summary,
                    learned: detail,
                    confidenceImproved: false,
                    output,
                    durationMs,
                    blocked: true,
                },
                evidence: [],
                retrievalSummary: null,
                resultPayload: JSON.stringify({ error: detail }),
            };
        }
    }
}

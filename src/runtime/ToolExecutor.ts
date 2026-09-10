import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import type {
    EvidenceItem,
    RetrievalSummary,
    ToolArguments,
    ToolInvocationRecord,
} from "@/runtime/contracts";
import { getToolStrategy, isMutatingTool } from "@/tools/registry";
import type { CapabilityContext } from "@/tools/types";
import type { DiscordToolName } from "@/shared/discordTools";
import type { DiscordToolResult } from "@/shared/appTypes";

export interface ToolExecutionResult {
    record: ToolInvocationRecord;
    evidence: EvidenceItem[];
    retrievalSummary: RetrievalSummary | null;
    resultPayload: string;
}

export interface ToolExecutionOptions {
    /** Read-only capabilities are bounded so one stuck dependency cannot hang a turn. */
    timeoutMs?: number;
}

function errorDetails(error: unknown): string {
    const value = error as {
        message?: unknown;
        code?: unknown;
        status?: unknown;
        rawError?: unknown;
    };
    const message = typeof value.message === "string" ? value.message : "Unknown tool error.";
    const code = value.code == null ? "" : ` (code: ${String(value.code)})`;
    const status = value.status == null ? "" : ` (status: ${String(value.status)})`;
    const raw = value.rawError == null ? "" : ` Raw: ${JSON.stringify(value.rawError)}`;
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
    public static async execute(
        toolName: DiscordToolName,
        args: ToolArguments,
        context: CapabilityContext,
        options: ToolExecutionOptions = {},
    ): Promise<ToolExecutionResult> {
        const startedAt = Date.now();
        try {
            const capability = CapabilityRegistry.get(toolName);
            const validation = capability.inputSchema.safeParse(args);
            if (!validation.success) {
                throw new Error(`Invalid arguments: ${validation.error.issues
                    .map((issue) => `${issue.path.join(".") || "input"}: ${issue.message}`)
                    .join("; ")}`);
            }
            const validatedArgs = validation.data as ToolArguments;
            const run = capability.run(context, validatedArgs);
            const timeoutMs = Math.max(1, options.timeoutMs ?? 0);
            const rawOutput = capability.sideEffectLevel === "none" && options.timeoutMs
                ? await withTimeout(run, timeoutMs)
                : await run;
            const outputValidation = capability.outputSchema.safeParse(rawOutput.data);
            if (!outputValidation.success) {
                throw new Error(`Invalid capability data: ${outputValidation.error.message}`);
            }
            const output: DiscordToolResult = {
                ...rawOutput,
                data: outputValidation.data,
            };
            const durationMs = Date.now() - startedAt;
            const strategy = getToolStrategy(toolName);
            const evidence = isMutatingTool(toolName) ? [] : strategy.extractEvidence(output);
            const retrievalSummary = isMutatingTool(toolName)
                ? null
                : strategy.extractRetrievalSummary?.(output) ?? null;
            const learned = evidence.map((item) => item.content).join(" | ") || output.summary;
            return {
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
            const durationMs = Date.now() - startedAt;
            const detail = `Tool ${toolName} failed: ${errorDetails(error)}`;
            const output = {
                tool: toolName,
                summary: `Failed to execute ${toolName}.`,
                data: null,
                errorMessage: detail,
            };
            return {
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

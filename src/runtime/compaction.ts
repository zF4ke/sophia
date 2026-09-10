import { ModelGateway, type ToolChatMessage } from "@/ai/ModelGateway";
import { SettingsService } from "@/app/SettingsService";
import { readModelProfiles } from "@/app/modelProfiles";
import type { ModelProfile, ModelProfileConfig } from "@/shared/appTypes";
import type { RuntimeTraceEvent } from "@/runtime/contracts";

/**
 * Tier-2 compaction: summarize the bulk of the conversation when
 * Tier-1 pruning isn't enough and prompt tokens are ≥ triggerFraction
 * of the context window.
 *
 * The summarizer replaces `messages[2..len-preserveTail]` with a single
 * system message, preserving the original system prompt (index 0),
 * first user message (index 1), and the last `preserveTail` messages.
 *
 * Note/plan tool calls are preserved verbatim in the summary because
 * they are load-bearing and cheap.
 */

/** Number of tail messages to keep verbatim (latest interaction). */
const PRESERVE_TAIL = 8;

/** Max tokens for the summary output. */
const SUMMARY_MAX_TOKENS = 4096;

/** Minimum middle-block size before compaction is worthwhile — avoids trivial summaries. */
const MIN_MIDDLE_BLOCK = 4;

/** Compaction tool names whose content is preserved in summaries. */
const LOAD_BEARING_TOOLS = new Set(["note_add", "note_list", "note_clear", "plan_update"]);

export interface CompactionInput {
    messages: ToolChatMessage[];
    promptTokens: number;
    contextWindow: number;
    traceEvents?: RuntimeTraceEvent[];
}

export interface CompactionResult {
    compacted: boolean;
    removedCount: number;
    summaryTokenEstimate: number;
}

/**
 * Should Tier-2 compaction run? True when prompt tokens ≥ triggerFraction
 * of the context window after Tier-1 pruning has already run.
 */
export function shouldCompact(promptTokens: number, contextWindow: number): boolean {
    const settings = SettingsService.load();
    let fraction = settings.compaction.triggerFraction;
    // For very large windows (GLM 5.3 Flash = 1M), avoid premature compaction on
    // moderately-sized prompts — cap effective threshold at 0.90 minimum for
    // windows > 500k to keep more context verbatim longer.
    if (contextWindow >= 500_000 && fraction < 0.90) {
        fraction = 0.90;
    }
    return promptTokens >= contextWindow * fraction;
}

/**
 * Resolve the model profile configured for compaction.
 * Falls back to the default profile if the configured one is missing.
 */
export function resolveCompactionProfile(): ModelProfile | null {
    const settings = SettingsService.load();
    const profileName = settings.compaction.summarizerModel;
    const config: ModelProfileConfig = readModelProfiles();
    const profile = config.profiles[profileName];
    if (profile) return profile;
    // Configured profile is missing — fall back to the default profile.
    const defaultProfile = config.profiles[config.defaultProfile];
    return defaultProfile ?? null;
}

/**
 * Extract the compactable region and build the summarisation prompt.
 */
function buildSummaryPrompt(messages: ToolChatMessage[]): {
    /** Messages to be summarised (the "middle" block). */
    middleBlock: ToolChatMessage[];
    /** Load-bearing tool calls extracted from the middle block. */
    preservedCallSummaries: string[];
} {
    // [0] = system prompt, [1] = first user message → always kept.
    const firstKept = 2;
    const lastKeptStart = Math.max(firstKept, messages.length - PRESERVE_TAIL);

    const middleBlock = messages.slice(firstKept, lastKeptStart);
    const preservedCallSummaries: string[] = [];

    for (const msg of middleBlock) {
        // Preserve note/plan tool calls verbatim.
        if (msg.role === "assistant" && "tool_calls" in msg && msg.tool_calls) {
            for (const tc of msg.tool_calls) {
                if (LOAD_BEARING_TOOLS.has(tc.function.name)) {
                    preservedCallSummaries.push(
                        `[${tc.function.name}] ${tc.function.arguments}`,
                    );
                }
            }
        }
    }

    return { middleBlock, preservedCallSummaries };
}

function serializeMiddleBlock(middleBlock: ToolChatMessage[]): string {
    return middleBlock
        .map((msg) => {
            if (msg.role === "tool") {
                return `[tool result] ${msg.content.slice(0, 800)}`;
            }
            if (msg.role === "assistant" && "tool_calls" in msg && msg.tool_calls) {
                const calls = msg.tool_calls
                    .map((tc) => `${tc.function.name}(${tc.function.arguments.slice(0, 200)})`)
                    .join("; ");
                return `[assistant tool_calls] ${calls}`;
            }
            return `[${msg.role}] ${("content" in msg ? msg.content : "")?.slice(0, 600) ?? ""}`;
        })
        .join("\n");
}

/**
 * Run Tier-2 compaction in place. Mutates `messages` array.
 *
 * Returns a result indicating whether compaction happened and how much
 * was removed.
 */
export async function compactMessages(
    input: CompactionInput,
): Promise<CompactionResult> {
    const { messages, promptTokens, contextWindow, traceEvents } = input;

    if (!shouldCompact(promptTokens, contextWindow)) {
        return { compacted: false, removedCount: 0, summaryTokenEstimate: 0 };
    }

    // Need at least system + user + meaningful middle + tail to compact.
    if (messages.length < 2 + PRESERVE_TAIL + MIN_MIDDLE_BLOCK) {
        return { compacted: false, removedCount: 0, summaryTokenEstimate: 0 };
    }

    const profile = resolveCompactionProfile();
    if (!profile) {
        return { compacted: false, removedCount: 0, summaryTokenEstimate: 0 };
    }

    const { middleBlock, preservedCallSummaries } = buildSummaryPrompt(messages);
    if (middleBlock.length === 0) {
        return { compacted: false, removedCount: 0, summaryTokenEstimate: 0 };
    }

    const transcript = serializeMiddleBlock(middleBlock);

    const preservedSection = preservedCallSummaries.length > 0
        ? `\n\nPreserved scratchpad calls (include verbatim in your summary):\n${preservedCallSummaries.join("\n")}`
        : "";

    const summarySystemPrompt = [
        "You are a conversation summariser for a Discord bot runtime.",
        "Summarise the following transcript of tool calls and results into a concise narrative.",
        "Preserve: user intent, key findings, tool names called, decisions made, any jumpLinks or message IDs mentioned, and all scratchpad (note/plan) entries.",
        "Do NOT invent jumpLinks or IDs that are not in the transcript.",
        "Output only the summary, no preamble.",
        `Max length: ~${SUMMARY_MAX_TOKENS} tokens.`,
        preservedSection,
    ].join("\n");

    try {
        const summaryText = await ModelGateway.generateText(
            [
                { role: "system", content: summarySystemPrompt },
                { role: "user", content: transcript },
            ],
            {
                profile,
                maxOutputTokens: SUMMARY_MAX_TOKENS,
                temperature: 0.2,
                traceContext: {
                    traceLabel: "compaction_tier2",
                    questionPreview: "Summarise conversation transcript",
                    traceEvents: traceEvents ? [...traceEvents] : undefined,
                },
            },
        );

        // Splice: remove middle block, insert summary message.
        const firstKept = 2;
        const lastKeptStart = Math.max(firstKept, messages.length - PRESERVE_TAIL);
        const removedCount = lastKeptStart - firstKept;

        const summaryMessage: ToolChatMessage = {
            role: "system",
            content: `<compaction_summary>\n${summaryText}\n</compaction_summary>`,
        };

        messages.splice(firstKept, removedCount, summaryMessage);

        const estimate = Math.ceil(summaryText.length / 4);

        if (traceEvents) {
            traceEvents.push({
                label: "compaction_tier2",
                detail: `Compacted ${removedCount} messages into summary (~${estimate} tokens). Model: ${profile.chatModel}`,
                timestamp: Date.now(),
            });
        }

        return { compacted: true, removedCount, summaryTokenEstimate: estimate };
    } catch (error) {
        if (traceEvents) {
            traceEvents.push({
                label: "compaction_tier2_error",
                detail: `Compaction failed: ${error instanceof Error ? error.message : String(error)}`,
                timestamp: Date.now(),
            });
        }
        return { compacted: false, removedCount: 0, summaryTokenEstimate: 0 };
    }
}

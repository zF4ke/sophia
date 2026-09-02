// Tier-0 input-side compaction.
//
// Runs once before the agent loop starts. Measures the assembled system
// prompt with a tokenizer, and if it's over `inputTriggerFraction * contextWindow`,
// summarises the bulky context fields (recent turns, channel context, prior
// evidence, prior tool runs) into a single compacted narrative. Personality,
// instructions, and the current question are always preserved verbatim.
//
// This complements Tier-2 in-loop compaction (src/runtime/compaction.ts),
// which only fires after the agent loop has already grown inside a single
// turn. Tier-0 addresses the pre-loop baseline: tiny social replies should
// not carry 32 evidence items and 15 channel messages into the prompt.

import { ModelGateway } from "@/ai/ModelGateway";
import { SettingsService } from "@/app/SettingsService";
import { resolveCompactionProfile } from "@/runtime/compaction";
import type { RuntimeTraceEvent } from "@/runtime/contracts";
import { countTokens } from "@/shared/tokenizer";

const SUMMARY_MAX_TOKENS = 1024;
const EMPTY_PLACEHOLDER = "None.";
const COMPACTED_PLACEHOLDER = "(compacted into Context Summary above)";

export interface InputCompactionFields {
    recentTurns: string;
    channelContext: string;
    priorEvidence: string;
    toolContext: string;
}

export interface InputCompactionInput {
    assembledPrompt: string;
    question: string;
    contextWindow: number;
    fields: InputCompactionFields;
    traceEvents?: RuntimeTraceEvent[];
}

export interface InputCompactionResult {
    compacted: boolean;
    promptTokensBefore: number;
    promptTokensAfter: number;
    summary: string | null;
    /** Drop-in replacements for the four fields when compacted. Null if no compaction. */
    fields: InputCompactionFields | null;
}

export function shouldCompactInput(
    promptTokens: number,
    contextWindow: number,
): boolean {
    let fraction = 0;
    try {
        fraction = SettingsService.load().compaction?.inputTriggerFraction ?? 0;
    } catch {
        fraction = 0;
    }
    if (!Number.isFinite(fraction) || fraction <= 0) return false;
    if (!Number.isFinite(contextWindow) || contextWindow <= 0) return false;
    // Adaptive: for large windows (>=500k) require at least 55% fill before
    // Tier-0 compaction to avoid summarizing away useful prior evidence on
    // small-to-medium prompts. For 1M window this means ~570k tokens.
    if (contextWindow >= 500_000 && fraction < 0.55) fraction = 0.55;
    return promptTokens >= contextWindow * fraction;
}

function hasSubstance(field: string): boolean {
    const trimmed = (field || "").trim();
    if (!trimmed) return false;
    if (trimmed === EMPTY_PLACEHOLDER) return false;
    if (/^no recent conversation turns\.?$/i.test(trimmed)) return false;
    if (/^no prior tool runs\.?$/i.test(trimmed)) return false;
    return true;
}

function buildTranscript(fields: InputCompactionFields): string {
    const parts: string[] = [];
    if (hasSubstance(fields.recentTurns)) {
        parts.push(`### Recent Turns\n${fields.recentTurns}`);
    }
    if (hasSubstance(fields.channelContext)) {
        parts.push(`### Recent Channel Messages\n${fields.channelContext}`);
    }
    if (hasSubstance(fields.priorEvidence)) {
        parts.push(`### Prior Evidence\n${fields.priorEvidence}`);
    }
    if (hasSubstance(fields.toolContext)) {
        parts.push(`### Prior Tool Runs\n${fields.toolContext}`);
    }
    return parts.join("\n\n");
}

export async function compactInput(
    input: InputCompactionInput,
): Promise<InputCompactionResult> {
    const { assembledPrompt, question, contextWindow, fields, traceEvents } = input;
    const promptTokensBefore = countTokens(assembledPrompt) + countTokens(question);

    if (!shouldCompactInput(promptTokensBefore, contextWindow)) {
        return {
            compacted: false,
            promptTokensBefore,
            promptTokensAfter: promptTokensBefore,
            summary: null,
            fields: null,
        };
    }

    const transcript = buildTranscript(fields);
    if (!transcript) {
        // Nothing bulky to compact — over-budget is caused by personality/instructions,
        // which we don't touch. Leave as-is.
        return {
            compacted: false,
            promptTokensBefore,
            promptTokensAfter: promptTokensBefore,
            summary: null,
            fields: null,
        };
    }

    const profile = resolveCompactionProfile();
    if (!profile) {
        return {
            compacted: false,
            promptTokensBefore,
            promptTokensAfter: promptTokensBefore,
            summary: null,
            fields: null,
        };
    }

    const summarySystemPrompt = [
        "You are a context summariser for a Discord bot (Sophia).",
        "Compress the following context into a concise narrative Sophia can use to answer the user's current question.",
        "",
        "Preserve: user intents, key facts, decisions, names/handles, channel names, message IDs or jumpLinks mentioned, recent conversational tone between specific speakers.",
        "Drop: verbatim quoting that is not load-bearing, redundant restatements, low-signal channel chatter unrelated to the current question.",
        "Do NOT invent jumpLinks, IDs, or facts not present in the transcript.",
        "Attribute statements to the correct speaker when speakers are distinguishable — never conflate two users.",
        "Output only the summary, no preamble, no headers.",
        `Max length: ~${SUMMARY_MAX_TOKENS} tokens.`,
    ].join("\n");

    const userPayload = [
        `Current question from user: ${question}`,
        "",
        "Context to compress:",
        transcript,
    ].join("\n");

    try {
        const summaryText = await ModelGateway.generateText(
            [
                { role: "system", content: summarySystemPrompt },
                { role: "user", content: userPayload },
            ],
            {
                profile,
                maxOutputTokens: SUMMARY_MAX_TOKENS,
                temperature: 0.2,
                traceContext: {
                    traceLabel: "compaction_tier0",
                    questionPreview: question.slice(0, 80),
                    traceEvents: traceEvents ? [...traceEvents] : undefined,
                },
            },
        );

        const cleanSummary = (summaryText || "").trim();
        if (!cleanSummary) {
            if (traceEvents) {
                traceEvents.push({
                    label: "compaction_tier0_empty",
                    detail: "Summariser returned empty output — skipping input compaction.",
                    timestamp: Date.now(),
                });
            }
            return {
                compacted: false,
                promptTokensBefore,
                promptTokensAfter: promptTokensBefore,
                summary: null,
                fields: null,
            };
        }

        // The first non-empty surviving field gets the summary prepended so the
        // compacted narrative lands in a stable, early section of the prompt.
        const replacedFields: InputCompactionFields = {
            recentTurns: hasSubstance(fields.recentTurns)
                ? `Context Summary:\n${cleanSummary}\n\n(original recent turns compacted above)`
                : fields.recentTurns,
            channelContext: hasSubstance(fields.channelContext)
                ? COMPACTED_PLACEHOLDER
                : fields.channelContext,
            priorEvidence: hasSubstance(fields.priorEvidence)
                ? COMPACTED_PLACEHOLDER
                : fields.priorEvidence,
            toolContext: hasSubstance(fields.toolContext)
                ? COMPACTED_PLACEHOLDER
                : fields.toolContext,
        };

        // If recentTurns had no substance, put the summary under the first
        // field that did — so the summary is never orphaned.
        if (!hasSubstance(fields.recentTurns)) {
            if (hasSubstance(fields.channelContext)) {
                replacedFields.channelContext = `Context Summary:\n${cleanSummary}`;
            } else if (hasSubstance(fields.priorEvidence)) {
                replacedFields.priorEvidence = `Context Summary:\n${cleanSummary}`;
            } else if (hasSubstance(fields.toolContext)) {
                replacedFields.toolContext = `Context Summary:\n${cleanSummary}`;
            }
        }

        const compactedApprox =
            countTokens(replacedFields.recentTurns) +
            countTokens(replacedFields.channelContext) +
            countTokens(replacedFields.priorEvidence) +
            countTokens(replacedFields.toolContext);
        const originalApprox =
            countTokens(fields.recentTurns) +
            countTokens(fields.channelContext) +
            countTokens(fields.priorEvidence) +
            countTokens(fields.toolContext);
        const delta = originalApprox - compactedApprox;
        const promptTokensAfter = Math.max(0, promptTokensBefore - delta);

        if (traceEvents) {
            traceEvents.push({
                label: "compaction_tier0",
                detail: `Compacted input context: ${promptTokensBefore} → ~${promptTokensAfter} tokens (saved ~${delta}). Model: ${profile.chatModel}`,
                timestamp: Date.now(),
            });
        }

        return {
            compacted: true,
            promptTokensBefore,
            promptTokensAfter,
            summary: cleanSummary,
            fields: replacedFields,
        };
    } catch (error) {
        if (traceEvents) {
            traceEvents.push({
                label: "compaction_tier0_error",
                detail: `Input compaction failed: ${error instanceof Error ? error.message : String(error)}`,
                timestamp: Date.now(),
            });
        }
        return {
            compacted: false,
            promptTokensBefore,
            promptTokensAfter: promptTokensBefore,
            summary: null,
            fields: null,
        };
    }
}

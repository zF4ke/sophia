import { ModelGateway } from "@/ai/ModelGateway";
import { SettingsService } from "@/app/SettingsService";
import { readModelProfiles } from "@/app/modelProfiles";
import { PromptRegistry } from "@/runtime/PromptRegistry";
import type { DiscordToolEvidenceRole } from "@/shared/discordTools";
import type { ModelProfile } from "@/shared/appTypes";
import type { ToolInvocationRecord } from "@/runtime/contracts";
import type { ToolEffect } from "@/tools/types";

/**
 * Stall / fake-promise detector.
 *
 * Uses a cheap LLM classifier to detect when a `finish` answer promises
 * action instead of delivering content. Language-agnostic — no regexes.
 *
 * Structural checks (evidence roles, write/destructive tool success)
 * are still deterministic and bypass the classifier entirely.
 */

export interface StallDetectionInput {
    /** The candidate answer the model is trying to send via `finish`. */
    answer: string;
    /**
     * Every tool record pushed this turn (includes blocked/failed).
     * Only records with `blocked !== true` and no `output.errorMessage`
     * count toward redemption.
     */
    toolHistoryThisTurn: ToolInvocationRecord[];
    /** Evidence roles produced by successful tool calls this turn. */
    evidenceRoles: Set<DiscordToolEvidenceRole>;
    /**
     * Lookup: tool name → its catalog effect ("read" | "write" |
     * "destructive"). Passed in so the detector stays decoupled from
     * the registry (easier to unit-test).
     */
    getToolEffect: (toolName: string) => ToolEffect | undefined;
    /**
     * True when `start_long_task` was granted this turn. The model
     * explicitly requested extended budget, so finishing without
     * evidence is always a stall — no promise phrase needed.
     */
    longTaskGranted?: boolean;
}

export interface StallDetectionResult {
    stalled: boolean;
    matchedPhrase?: string;
}

/**
 * Evidence roles that redeem a promise about searching/reading Discord.
 */
const PRODUCTIVE_EVIDENCE_ROLES: ReadonlySet<DiscordToolEvidenceRole> = new Set([
    "message_evidence",
    "history_evidence",
    "semantic_evidence",
    "live_evidence",
]);

// ── AI-based promise classifier ─────────────────────────────────────

interface ClassifierResult {
    stall: boolean;
}

/**
 * Resolve the model profile used for the stall classifier.
 * Uses the compaction summarizer model from settings.
 */
function resolveClassifierProfile(): ModelProfile | null {
    const settings = SettingsService.load();
    const profileName = settings.compaction.summarizerModel;
    const config = readModelProfiles();
    return config.profiles[profileName] ?? config.profiles[config.defaultProfile] ?? null;
}

/**
 * Call a cheap LLM to classify whether `answer` is a stall/promise.
 * Fail-open: returns `false` (not a stall) on any error.
 */
async function classifyPromise(answer: string): Promise<boolean> {
    try {
        const systemPrompt = PromptRegistry.load("runtime/stall_classifier");
        const result = await ModelGateway.generateJson<ClassifierResult>(
            [
                { role: "system", content: systemPrompt },
                { role: "user", content: answer },
            ],
            { stall: false },
            {
                profile: resolveClassifierProfile() ?? undefined,
                temperature: 0,
                maxOutputTokens: 32,
                traceContext: { traceLabel: "stall_classifier" },
            },
        );
        return result.stall === true;
    } catch {
        return false;
    }
}

function isProductiveRecord(
    record: ToolInvocationRecord,
    getToolEffect: (toolName: string) => ToolEffect | undefined,
): boolean {
    if (record.blocked) return false;
    if (record.output?.errorMessage) return false;
    const effect = getToolEffect(record.tool);
    if (effect === "write" || effect === "destructive") return true;
    return false;
}

/**
 * Detect whether the candidate `finish` answer is an empty promise.
 *
 * Returns `stalled = true` when:
 *   A. the AI classifier flags the answer as a promise AND no
 *      productive work was done, OR
 *   B. longTaskGranted is true AND no productive evidence was produced
 *      (the model asked for extended budget then bailed).
 *
 * Structural redemption (evidence roles, write/destructive tools)
 * short-circuits before the classifier is called, keeping latency low.
 */
export async function detectStallPromise(input: StallDetectionInput): Promise<StallDetectionResult> {
    const { answer, toolHistoryThisTurn, evidenceRoles, getToolEffect, longTaskGranted } = input;
    if (!answer || answer.trim().length === 0) {
        return { stalled: false };
    }

    const ranProductiveEvidence = [...evidenceRoles].some((role) =>
        PRODUCTIVE_EVIDENCE_ROLES.has(role),
    );
    const ranActionTool = toolHistoryThisTurn.some((record) =>
        isProductiveRecord(record, getToolEffect),
    );

    // ── Path B: long-task grant scenarios ──
    if (longTaskGranted) {
        const isPromise = await classifyPromise(answer);
        // B1: promise + long task → always stalled (evidence doesn't redeem)
        if (isPromise) {
            return { stalled: true, matchedPhrase: "ai_classified_promise" };
        }
        // B2: no promise, but ran prep tools without evidence → stalled
        if (!ranProductiveEvidence && !ranActionTool && toolHistoryThisTurn.length > 0) {
            return { stalled: true, matchedPhrase: "long_task_without_evidence" };
        }
        return { stalled: false };
    }

    // ── Path A: normal turns ──
    // Short-circuit: productive work done → not stalled, skip classifier.
    if (ranProductiveEvidence || ranActionTool) {
        return { stalled: false };
    }

    const isPromise = await classifyPromise(answer);
    if (isPromise) {
        return { stalled: true, matchedPhrase: "ai_classified_promise" };
    }

    return { stalled: false };
}

// ── Doom-loop detector ──────────────────────────────────────────────

/**
 * Tracks the last N tool calls for doom-loop detection.
 * Mutable state — owned by the Runtime per turn.
 */
export class DoomLoopDetector {
    /** Sliding window of the last `windowSize` (tool, argsHash) pairs. */
    private readonly window: string[] = [];
    private readonly windowSize: number;
    private readonly repeatThreshold: number;
    private nudgeCount = 0;

    constructor(windowSize = 5, repeatThreshold = 3) {
        this.windowSize = windowSize;
        this.repeatThreshold = repeatThreshold;
    }

    /**
     * Record a tool call. Returns a nudge message if the same
     * (tool, args) pair has been called `repeatThreshold` times
     * consecutively in the window.
     */
    recordCall(toolName: string, argsHash: string): DoomLoopResult {
        const key = `${toolName}:${argsHash}`;
        this.window.push(key);
        if (this.window.length > this.windowSize) {
            this.window.shift();
        }

        // Count consecutive identical calls from the tail.
        let consecutive = 0;
        for (let i = this.window.length - 1; i >= 0; i--) {
            if (this.window[i] === key) consecutive++;
            else break;
        }

        if (consecutive >= this.repeatThreshold) {
            this.nudgeCount++;
            if (this.nudgeCount >= 2) {
                return {
                    action: "force_finish",
                    message: `You've called ${toolName} with identical args ${consecutive} times and already received a warning. Finishing now with available data.`,
                };
            }
            return {
                action: "nudge",
                message: `You've called ${toolName} with identical args ${consecutive} times. Either change your approach, use note_add to record progress, or call finish with your answer.`,
            };
        }

        return { action: "ok" };
    }

    getNudgeCount(): number {
        return this.nudgeCount;
    }
}

export interface DoomLoopResult {
    action: "ok" | "nudge" | "force_finish";
    message?: string;
}

// ── Progress-required detector (long-task only) ─────────────────────

/** Tools that count as "progress" in the progress-required check. */
const PROGRESS_TOOLS = new Set(["note_add", "note_list", "plan_update"]);

/**
 * Tracks whether the model is making progress during long tasks.
 * "Progress" = note_add, plan_update, or new evidence retrieved.
 */
export class ProgressTracker {
    private callsSinceProgress = 0;
    private readonly threshold: number;

    constructor(threshold = 5) {
        this.threshold = threshold;
    }

    /**
     * Record a tool call. Call this AFTER the tool executes.
     * @param toolName The tool that was called.
     * @param producedEvidence Whether the call produced new evidence items.
     * @returns A nudge message if the model has gone too long without progress.
     */
    recordCall(toolName: string, producedEvidence: boolean): ProgressResult {
        if (PROGRESS_TOOLS.has(toolName) || producedEvidence) {
            this.callsSinceProgress = 0;
            return { stalled: false };
        }

        this.callsSinceProgress++;
        if (this.callsSinceProgress >= this.threshold) {
            this.callsSinceProgress = 0; // Reset after nudge.
            return {
                stalled: true,
                message: `You've made ${this.threshold} tool calls without recording any findings (note_add / plan_update) or retrieving new evidence. Record your progress or change approach.`,
            };
        }

        return { stalled: false };
    }
}

export interface ProgressResult {
    stalled: boolean;
    message?: string;
}

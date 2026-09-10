import type { DiscordToolEvidenceRole } from "@/shared/discordTools";
import type { ToolInvocationRecord } from "@/runtime/contracts";
import type { ToolEffect } from "@/tools/types";

/**
 * Stall / fake-promise detector.
 *
 * Detects a small set of obvious deferrals such as "let me check" and
 * "vou verificar". This check stays local because adding a model request to
 * validate every direct answer costs latency and creates another failure path.
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

const STALL_PHRASES: ReadonlyArray<{ label: string; pattern: RegExp }> = [
    {
        label: "english_deferred_action",
        pattern: /\b(?:let me|i(?:'|’)ll|i will)\s+(?:check|look|search|investigate|verify|review|collect|process|create|delete|edit|send|move)\b/i,
    },
    {
        label: "english_wait",
        pattern: /\b(?:give me (?:a )?moment|one moment|hold on|working on it|still (?:checking|searching|collecting|processing)|almost done|i(?:'|’)ll get back to you)\b/i,
    },
    {
        label: "portuguese_deferred_action",
        pattern: /\b(?:vou|irei)\s+(?:verificar|procurar|buscar|pesquisar|analisar|checar|coletar|processar|criar|apagar|deletar|editar|enviar|mover|limpar|dar uma olhada)\b/i,
    },
    {
        label: "portuguese_in_progress",
        pattern: /\b(?:estou|ainda estou)\s+(?:verificando|procurando|buscando|pesquisando|analisando|checando|coletando|processando|começando)\b/i,
    },
    {
        label: "portuguese_wait",
        pattern: /\b(?:só um momento|um minutinho|já volto|te aviso quando|assim que terminar)\b/i,
    },
];

function findStallPhrase(answer: string): string | undefined {
    return STALL_PHRASES.find(({ pattern }) => pattern.test(answer))?.label;
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
 *   A. the answer contains an obvious deferral AND no productive work
 *      was done, OR
 *   B. longTaskGranted is true AND no productive evidence was produced
 *      (the model asked for extended budget then bailed).
 *
 * Structural redemption from evidence or a completed mutation wins over
 * wording. A useful answer may legitimately say what it checked.
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
        const matchedPhrase = findStallPhrase(answer);
        // A long-task answer that still promises future work has not completed.
        if (matchedPhrase) {
            return { stalled: true, matchedPhrase };
        }
        // B2: no promise, but ran prep tools without evidence → stalled
        if (!ranProductiveEvidence && !ranActionTool && toolHistoryThisTurn.length > 0) {
            return { stalled: true, matchedPhrase: "long_task_without_evidence" };
        }
        return { stalled: false };
    }

    // ── Path A: normal turns ──
    // Productive work redeems promise-like wording.
    if (ranProductiveEvidence || ranActionTool) {
        return { stalled: false };
    }

    const matchedPhrase = findStallPhrase(answer);
    return matchedPhrase ? { stalled: true, matchedPhrase } : { stalled: false };
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

    constructor(windowSize = 6, repeatThreshold = 4) {
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

    constructor(threshold = 8) {
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

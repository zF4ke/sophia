import { Runtime } from "@/runtime/Runtime";
import type { CorpusTaskState } from "@/runtime/corpusTaskState";
import { DiscordMemoryService, type RequestGoalRecord } from "@/memory/DiscordMemoryService";
import type { RuntimeAnswer, TurnInput } from "@/runtime/contracts";

/**
 * Auto-continue loop for long-running work.
 *
 * When a turn finishes while any goal in this thread is still open (task
 * framed with goal_open, checkpointed to notes), the runtime spawns the next
 * turn itself instead of waiting for the user to say "continua". The loop is
 * driven entirely by persisted tool-call state — no answer-text matching:
 *
 * - goals are read from the `request_goals` table (open / in_progress count
 *   as unfinished; done / cancelled do not),
 * - a leg that calls `goal_done`/`goal_update` on every remaining open goal
 *   closes the chain,
 * - each turn also auto-opens a goal for large corpus tasks it never framed,
 *   so legacy behavior without explicit goals still chains.
 *
 * The chain is bounded and safe:
 * - each continuation reuses the same conversation input with a synthetic
 *   "continue" question,
 * - progress is posted into the channel between legs so the user sees life,
 * - legs never re-chain (trigger is auto_continue), so depth is exactly 1,
 * - hard cap (MAX_CONTINUATIONS) bounds the loop.
 */

const MAX_CONTINUATIONS = 20;
/** Idle gap between continuation legs so Discord typing/progress stays sane. */
const INTER_TURN_DELAY_MS = 3_000;

function isUnfinishedStatus(status: string): boolean {
    return status === "open" || status === "in_progress";
}

export interface AutoContinueResult {
    answer: RuntimeAnswer;
    legs: number;
    stoppedBecause: "complete" | "model_done" | "blocked" | "cap_reached" | "error";
}

/**
 * Runs the auto-continue chain after an initial answer. Returns the final
 * answer (the last leg's answer) plus how many continuation legs ran.
 * The caller is responsible for having already persisted/sent the first
 * answer; this module sends progress notes for each leg itself.
 */
export async function runAutoContinue(
    firstAnswer: RuntimeAnswer,
    input: TurnInput,
    corpusTaskState: CorpusTaskState | null,
    maxLegs: number = MAX_CONTINUATIONS,
): Promise<AutoContinueResult> {
    let answer = firstAnswer;
    let legs = 0;

    const progressChannel = input.currentChannelId;

    for (let leg = 1; leg <= maxLegs; leg++) {
        const openGoals = await listOpenGoals(input, firstAnswer.threadId);
        if (openGoals.length === 0) {
            return { answer, legs, stoppedBecause: "complete" };
        }

        const goalLines = openGoals
            .map((goal) => `- #${goal.seq} [${goal.status}] ${goal.body.slice(0, 80)}`)
            .join("\n");

        // One human status line per leg, rendered through a single tiny
        // model call (statusLine module) — never raw goal bodies, never
        // string-sliced, so it cannot duplicate or cut off mid-word.
        try {
            const { renderStatusLine } = await import("@/runtime/statusLine");
            const statusLine = await renderStatusLine(openGoals, leg);
            if (statusLine) {
                const guild = input.guild;
                const channel = guild && progressChannel ? guild.channels.cache.get(progressChannel) : null;
                if (channel && "send" in channel) {
                    await (channel as { send: (o: unknown) => Promise<unknown> }).send({
                        content: statusLine,
                    });
                }
            }
        } catch {
            // Best-effort.
        }

        await new Promise((resolve) => setTimeout(resolve, INTER_TURN_DELAY_MS));

        const legInput: TurnInput = {
            ...input,
            trigger: "auto_continue",
            question:
                `CONTINUE (auto): o usuário já aprovou continuação automática e ${openGoals.length} objetivo(s) seguem abertos:\n` +
                `${goalLines}\n\n` +
                `Retome do último checkpoint salvo nesta thread (note_list com include_thread_history) e continue exatamente esses objetivos abertos. ` +
                `Não pergunte — o usuário NÃO vai responder. ` +
                `Feche cada objetivo entregue com goal_done (ou goal_update → blocked se realmente travou). ` +
                `Só chame finish quando não restar nenhum objetivo aberto.`,
        };

        let legAnswer: RuntimeAnswer;
        try {
            legAnswer = await Runtime.answer(legInput);
        } catch (error) {
            console.error("[auto-continue] leg failed:", (error as Error).message);
            return { answer, legs, stoppedBecause: "error" };
        }

        legs += 1;
        answer = legAnswer;

        // Don't post intermediate leg answers — let the tool calls flow
        // continuously. Only the final answer (returned to the caller) will
        // be sent. This keeps the auto-continue invisible: same status line,
        // same "Running retrieve_messages…" indicators, as if nothing happened.

        // Empty answer with no new tool runs means the leg died: don't burn
        // more turns on a broken pipeline.
        if (!legAnswer.answer.trim() && legAnswer.toolRuns.length === 0) {
            return { answer: legAnswer, legs, stoppedBecause: "error" };
        }
    }

    return { answer, legs, stoppedBecause: "cap_reached" };
}

async function listOpenGoals(input: TurnInput, threadId: string): Promise<RequestGoalRecord[]> {
    try {
        const goals = await DiscordMemoryService.listRequestGoals({
            requestId: "*",
            threadId,
            includeThreadHistory: true,
        });
        return goals.filter((goal) => isUnfinishedStatus(goal.status));
    } catch {
        return [];
    }
}

/** Convenience: register the synthetic continuation request as a tool-run-free turn. */
export function newContinuationRequest(original: TurnInput, legNumber: number): TurnInput {
    return {
        ...original,
        question: `CONTINUE (auto, parte ${legNumber})`,
    };
}

export const AUTO_CONTINUE_CAP = MAX_CONTINUATIONS;

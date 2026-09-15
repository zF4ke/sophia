import { Runtime } from "@/runtime/Runtime";
import { getWorkingState } from "@/runtime/tasks/workingState";
import type { CorpusTaskState } from "@/runtime/corpusTaskState";
import type { RequestGoalRecord } from "@/memory/DiscordMemoryService";
import type { RuntimeAnswer, TurnInput } from "@/runtime/contracts";

/** Transitional continuation adapter, retained until persistent tasks replace it.
 * There is no default leg cap. Runtime-owned execution state carries explicit
 * limits and cancellation across legs; failed or paused results do not restart.
 */


function isUnfinishedStatus(status: string): boolean {
    return status === "open" || status === "in_progress";
}

export interface AutoContinueResult {
    answer: RuntimeAnswer;
    legs: number;
    stoppedBecause: "complete" | "model_done" | "blocked" | "cap_reached" | "error" | "cancelled";
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
    maxLegs: number = Number.POSITIVE_INFINITY,
): Promise<AutoContinueResult> {
    let answer = firstAnswer;
    let legs = 0;

    for (let leg = 1; leg <= maxLegs; leg++) {
        if (input.execution?.signal.aborted) return {
            answer: { ...answer, outcome: "cancelled", answer: "Execução interrompida. As ações já concluídas mantêm-se." },
            legs, stoppedBecause: "cancelled",
        };
        if (answer.outcome && answer.outcome !== "completed") return { answer, legs, stoppedBecause: answer.outcome === "failed" ? "error" : "blocked" };
        const openGoals = await listOpenGoals(input, firstAnswer.threadId);
        if (openGoals.length === 0) {
            return { answer, legs, stoppedBecause: "complete" };
        }

        try {
            if (input.progressNotifier) {
                await input.progressNotifier(`A continuar o pedido. Objetivos pendentes: ${openGoals.length}.`);
            }
        } catch {
            // Best-effort.
        }

        await new Promise<void>(resolve => setImmediate(resolve));

        const legInput: TurnInput = {
            ...input,
            trigger: "auto_continue",
            continuationContext: { goals: openGoals.map(goal => ({ id: goal.seq, status: goal.status, body: goal.body })), previousAnswer: answer.answer },
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
        if (legAnswer.toolRuns.length === 0) {
            const remaining = await listOpenGoals(input, firstAnswer.threadId);
            return { answer: remaining.length > 0 && legAnswer.outcome !== "failed"
                ? { ...legAnswer, outcome: "paused" } : legAnswer,
                legs, stoppedBecause: legAnswer.outcome === "failed" ? "error" : remaining.length > 0 ? "blocked" : "complete" };
        }
    }

    return { answer: { ...answer, outcome: "paused" }, legs, stoppedBecause: "cap_reached" };
}

async function listOpenGoals(input: TurnInput, threadId: string): Promise<RequestGoalRecord[]> {
    try {
        const state = await getWorkingState({ taskId: input.taskId, actorId: input.user.id, currentChannelId: input.currentChannelId, guild: input.guild });
        const goals = await state.listRequestGoals({
            requestId: "*",
            threadId,
            includeThreadHistory: true,
        });
        return goals.filter((goal) => isUnfinishedStatus(goal.status));
    } catch (error) {
        if (input.taskId) throw error;
        return [];
    }
}

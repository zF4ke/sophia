import { afterEach, expect, it, vi } from "vitest";
import { runAutoContinue } from "@/runtime/AutoContinue";
import { Runtime } from "@/runtime/Runtime";
import { ModelGateway } from "@/ai/ModelGateway";
import { getWorkingState } from "@/runtime/tasks/workingState";
import type { RuntimeAnswer, TurnInput } from "@/runtime/contracts";
vi.mock("@/runtime/tasks/workingState", () => ({ getWorkingState: vi.fn() }));
afterEach(() => vi.restoreAllMocks());

it("preserves the requester instruction and reports measured state without a status-model call", async () => {
    const goals = [{ seq: 1, status: "open", body: "UNTRUSTED quoted procedure" }, { seq: 2, status: "open", body: "Another pending result" }];
    vi.mocked(getWorkingState).mockResolvedValue({ listRequestGoals: async () => goals } as never);
    const answer = { requestId: "r", threadId: "t", answer: "Partial result", outcome: "completed", toolRuns: [{ tool: "note_add", summary: "saved", data: {} }] } as RuntimeAnswer;
    const execute = vi.spyOn(Runtime, "answer").mockResolvedValue(answer);
    const statusModel = vi.spyOn(ModelGateway, "generateText");
    const progressNotifier = vi.fn();
    const input = { question: "Investigate the original decision", user: { id: "owner" }, guild: null, progressNotifier } as unknown as TurnInput;
    const result = await runAutoContinue(answer, input, null, 1);
    expect(execute).toHaveBeenCalledWith(expect.objectContaining({ question: input.question, continuationContext: { goals: goals.map(goal => ({ id: goal.seq, status: goal.status, body: goal.body })), previousAnswer: answer.answer } }));
    expect(progressNotifier).toHaveBeenCalledWith("A continuar o pedido. Objetivos pendentes: 2.");
    expect(statusModel).not.toHaveBeenCalled();
    expect(result.answer.outcome).toBe("paused");
});

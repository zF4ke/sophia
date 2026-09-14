import { afterEach, describe, expect, it, vi } from "vitest";
import { Runtime } from "@/runtime/Runtime";
import { runAutoContinue } from "@/runtime/AutoContinue";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { RuntimeAnswer, TurnInput } from "@/runtime/contracts";

vi.mock("@/runtime/Runtime", () => ({ Runtime: { answer: vi.fn() } }));

const answer: RuntimeAnswer = { requestId: "r", threadId: "t", answer: "Progress saved.",
    outcome: "completed", citations: [], classification: { mode: "direct_answer", reason: "test" },
    toolRuns: [{ tool: "note_add", summary: "saved", data: {} }], confidence: "best_effort" };
const input = { question: "Continue", user: { id: "owner" }, guild: null, trigger: "talk",
    conversation: { key: "t", kind: "channel", trigger: "talk" } } as TurnInput;

afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

describe("continuation lifecycle", () => {
    it("continues past the former 20-leg cap and stops when goals close", async () => {
        vi.useFakeTimers();
        let legs = 0;
        vi.spyOn(DiscordMemoryService, "listRequestGoals").mockImplementation(async () =>
            legs < 21 ? [{ seq: 1, status: "in_progress", body: "Work" }] as never : []);
        vi.mocked(Runtime.answer).mockImplementation(async () => { legs++; return answer; });
        const pending = runAutoContinue(answer, input, null);
        await vi.runAllTimersAsync();
        expect(await pending).toMatchObject({ legs: 21, stoppedBecause: "complete" });
    });

    it("does not retry a provider failure just because open goals remain", async () => {
        const model = vi.mocked(Runtime.answer);
        model.mockClear();
        expect(await runAutoContinue({ ...answer, outcome: "failed", toolRuns: [] }, input, null))
            .toMatchObject({ legs: 0, stoppedBecause: "error" });
        expect(model).not.toHaveBeenCalled();
    });
});

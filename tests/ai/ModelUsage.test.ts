import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelUsage, type ModelUsageRecord } from "@/ai/ModelUsage";
import { taskStore } from "@/runtime/tasks/TaskStore";

afterEach(() => vi.restoreAllMocks());
const profile = { chatModel: "model", pricing: { inputPerMillionUsd: 2, outputPerMillionUsd: 4 } };
describe("provider usage accounting", () => {
    it("keeps concurrent task and background ownership distinct and records failures as unknown usage", async () => {
        const records: ModelUsageRecord[] = [];
        vi.spyOn(taskStore, "recordModelUsage").mockImplementation(async record => { records.push(record); });
        let release!: () => void;
        const held = new Promise<void>(resolve => { release = resolve; });
        const first = ModelUsage.scope({ taskId: "first", actorId: "alice" }, () => ModelUsage.measure(profile, "compaction", async () => {
            await held;
            return { usage: { prompt_tokens: 100, completion_tokens: 20 } };
        }));
        const failure = Object.assign(new Error("private provider payload"), { status: 429 });
        await expect(ModelUsage.scope({ taskId: "second", actorId: "bob", job: "dream" }, () => ModelUsage.measure(profile, "dreaming", async () => { throw failure; }))).rejects.toBe(failure);
        release();
        await first;
        expect(records).toEqual(expect.arrayContaining([
            expect.objectContaining({ taskId: "first", actorId: "alice", purpose: "compaction", status: "returned", promptTokens: 100, completionTokens: 20, estimatedCostUsd: 0.00028 }),
            expect.objectContaining({ taskId: "second", actorId: "bob", job: "dream", status: "failed", errorCode: "429", promptTokens: null, estimatedCostUsd: null }),
        ]));
        expect(JSON.stringify(records)).not.toContain("private provider payload");
        await ModelUsage.measure(profile, "unowned maintenance", async () => ({}));
        expect(records[2].taskId).toBeUndefined();
        expect(records[2].promptTokens).toBeNull();
    });
});

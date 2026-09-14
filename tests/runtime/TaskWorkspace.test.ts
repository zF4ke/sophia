import path from "node:path";
import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { TaskStore } from "@/runtime/tasks/TaskStore";

const stores: TaskStore[] = [];
afterEach(async () => { for (const store of stores.splice(0)) await store.close(); });
const owner = { actorId: "owner", guildId: "g1", channelId: "c1", conversationId: "shared-channel", objective: "Work" };
const leg = { requestId: "leg-one", threadId: "shared-channel" };
function open(filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`), session = "session-one") {
    const store = new TaskStore(filename, session);
    stores.push(store);
    return store;
}

describe("task working state", () => {
    it("invalidates derived notes and blocks late writes from the deleted-source turn, while allowing fresh work", async () => {
        const store = open();
        const taskId = await store.create(owner);
        const state = await store.workspace(taskId, owner.actorId, owner.channelId, owner.guildId);
        await store.recordEvidenceSources({ ...owner, taskId, requestId: leg.requestId }, [{ messageId: "source", channelId: "source-channel" }]);
        await state.addRequestNote({ ...leg, kind: "note", label: null, body: "Derived private fact" });
        await state.upsertRequestPlan({ ...leg, body: "Derived plan" });
        await state.addRequestGoal({ ...leg, label: null, body: "Derived goal" });
        await store.invalidateCorpusMessage("source");
        expect(await state.listRequestNotes({ ...leg })).toEqual([]);
        expect(await state.listRequestGoals({ ...leg })).toEqual([]);
        await expect(state.addRequestNote({ ...leg, kind: "note", label: null, body: "Late stale fact" })).rejects.toThrow("source was deleted");
        await expect(state.upsertRequestPlan({ ...leg, body: "Late stale plan" })).rejects.toThrow("source was deleted");
        await store.recordEvidenceSources({ ...owner, taskId, requestId: "fresh" }, []);
        await state.addRequestNote({ ...leg, requestId: "fresh", kind: "note", label: null, body: "Fresh work" });
        expect(await state.listRequestNotes({ ...leg })).toHaveLength(1);
        expect(await store.requestSources(leg.requestId)).toBeNull();
    });
    it("isolates tasks in the same channel, including tasks belonging to the same owner", async () => {
        const store = open();
        const first = await store.create(owner);
        const second = await store.create(owner);
        const a = await store.workspace(first, "owner", "c1", "g1");
        const b = await store.workspace(second, "owner", "c1", "g1");
        await a.addRequestNote({ ...leg, kind: "note", label: null, body: "Only first task" });
        await a.upsertRequestPlan({ ...leg, body: "First plan" });
        const goal = await a.addRequestGoal({ ...leg, label: null, body: "First goal" });
        expect(await b.listRequestNotes({ ...leg, includeThreadHistory: true })).toEqual([]);
        expect(await b.getRequestPlan(leg.requestId)).toBeNull();
        expect(await b.listRequestGoals({ ...leg, includeThreadHistory: true })).toEqual([]);
        await expect(b.updateRequestGoal({ requestId: leg.requestId, seq: goal.seq, status: "done" })).rejects.toThrow("does not belong");
        await b.clearRequestNotes({ requestId: leg.requestId });
        expect((await a.listRequestNotes({ requestId: "later-leg", kind: "note" }))[0].body).toBe("Only first task");
        await expect(store.workspace(first, "other", "c1", "g1")).rejects.toThrow("not owned");
        expect(await store.snapshot(first, "other", "c1", "g1")).toBeNull();
    });

    it("keeps stable note IDs and increments plan versions across continuation legs", async () => {
        const store = open();
        const id = await store.create(owner);
        const state = await store.workspace(id, "owner", "c1", "g1");
        const notes = await Promise.all(Array.from({ length: 20 }, (_, index) => state.addRequestNote({ ...leg, kind: "note", label: "keep", body: String(index) })));
        expect(new Set(notes.map(note => note.seq)).size).toBe(20);
        expect(await state.upsertRequestPlan({ ...leg, body: "v1" })).toEqual({ version: 1 });
        expect(await state.upsertRequestPlan({ ...leg, requestId: "leg-two", body: "v2" })).toEqual({ version: 2 });
        expect(await state.upsertRequestPlan({ ...leg, requestId: "leg-three", body: "v3" })).toEqual({ version: 3 });
        expect(await state.getRequestPlan("leg-four")).toBe("v3");
        expect(await state.countRequestNotes({ requestId: "leg-four", kind: "note" })).toBe(20);
        await state.clearRequestNotes({ requestId: "leg-four", kind: "note", label: "keep" });
        expect(await state.getRequestPlan("leg-four")).toBe("v3");
        const next = await state.addRequestNote({ ...leg, kind: "note", label: null, body: "New" });
        expect(next.seq).toBeGreaterThan(Math.max(...notes.map(note => note.seq)));
    });

    it("preserves working state across index resets and exposes it read-only after interruption", async () => {
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        const filename = path.join(AppPaths.storageRoot, "tasks.sqlite");
        const first = open(filename);
        const id = await first.create(owner);
        const state = await first.workspace(id, "owner", "c1", "g1");
        await state.upsertRequestPlan({ ...leg, body: "Check the remaining sources" });
        await state.addRequestNote({ ...leg, kind: "note", label: "evidence", body: "Source A inspected" });
        await state.addRequestGoal({ ...leg, label: null, body: "Inspect source B" });
        const { RuntimeStorageService } = await import("@/runtime/storage/RuntimeStorageService");
        await RuntimeStorageService.resetAllRuntimeData();
        await first.close();
        const next = open(filename, "session-two");
        expect(await next.snapshot(id, "owner", "c1", "g1")).toMatchObject({ plan: "Check the remaining sources",
            notes: [{ body: "Source A inspected" }], goals: [{ body: "Inspect source B", status: "open" }] });
        await expect(next.workspace(id, "owner", "c1", "g1")).rejects.toThrow("not owned");
    });
});

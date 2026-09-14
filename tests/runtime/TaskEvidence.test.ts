import path from "node:path";
import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { TaskStore } from "@/runtime/tasks/TaskStore";
import type { ToolInvocationRecord } from "@/runtime/contracts";

const stores: TaskStore[] = [];
afterEach(async () => { for (const store of stores.splice(0)) await store.close(); });
const owner = { actorId: "owner", channelId: "c1", guildId: "g1", conversationId: "conversation", objective: "Read history" };
const record: ToolInvocationRecord = { tool: "retrieve_messages", arguments: { channel_ids: ["c1"] }, summary: "Read m1",
    learned: "A source", confidenceImproved: true, durationMs: 10,
    output: { tool: "retrieve_messages", summary: "Read m1", data: { messageId: "m1", jumpLink: "https://discord.com/channels/g1/c1/m1", cursor: { before: "m1" } } } };
function open(filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`), session = "one") {
    const store = new TaskStore(filename, session); stores.push(store); return store;
}

describe("task evidence", () => {
    it("removes persisted source text when a referenced message is deleted", async () => {
        const store = open();
        const taskId = await store.create(owner);
        await store.recordToolRun({ ...owner, taskId, requestId: "leg", invocationId: "call", record });
        await store.invalidateCorpusMessage("m1");
        const runs = await store.toolRuns(taskId, "owner", "c1", "g1");
        expect(runs[0].blocked).toBe(true);
        expect(JSON.stringify(runs)).not.toContain("A source");
        expect(runs[0].output.data).toBeNull();
        await store.recordToolRun({ ...owner, taskId, requestId: "leg", invocationId: "late-call", record });
        expect(JSON.stringify(await store.toolRuns(taskId, "owner", "c1", "g1"))).not.toContain("A source");
    });
    it("keeps evidence within its task and authenticated owner/location", async () => {
        const store = open();
        const taskId = await store.create(owner);
        const neighbor = await store.create(owner);
        const input = { ...owner, taskId, requestId: "leg-1", invocationId: "call-1", record };
        await store.recordToolRun(input);
        expect(await store.toolRuns(taskId, "owner", "c1", "g1")).toEqual([record]);
        expect(await store.toolRuns(neighbor, "owner", "c1", "g1")).toEqual([]);
        expect(await store.toolRuns(taskId, "other", "c1", "g1")).toEqual([]);
        expect(await store.toolRuns(taskId, "owner", "c2", "g1")).toEqual([]);
        expect(await store.toolRuns(taskId, "owner", "c1", "g2")).toEqual([]);
        await expect(store.recordToolRun({ ...input, actorId: "other" })).rejects.toThrow();
        await expect(store.recordToolRun(input)).rejects.toThrow();
    });

    it("preserves full results across continuation legs, context limits, resets and restart", async () => {
        const filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`);
        const first = open(filename);
        const taskId = await first.create(owner);
        await first.recordToolRun({ ...owner, taskId, requestId: "leg-1", invocationId: "call-1", record });
        const later = { ...record, summary: "Next page" };
        await first.recordToolRun({ ...owner, taskId, requestId: "leg-2", invocationId: "call-2", record: later });
        expect(await first.toolRuns(taskId, "owner", "c1", "g1", 1)).toEqual([later]);
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        const { RuntimeStorageService } = await import("@/runtime/storage/RuntimeStorageService");
        await RuntimeStorageService.resetAllRuntimeData();
        await first.close();
        const next = open(filename, "two");
        expect(await next.toolRuns(taskId, "owner", "c1", "g1")).toEqual([later, record]);
        await expect(next.recordToolRun({ ...owner, taskId, requestId: "leg-3", invocationId: "call-3", record })).rejects.toThrow();
    });
});

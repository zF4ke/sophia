import path from "node:path";
import fs from "node:fs";
import { randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { TaskStore } from "@/runtime/tasks/TaskStore";

const stores: TaskStore[] = [];
afterEach(async () => { for (const store of stores.splice(0)) await store.close(); });
const request = { actorId: "owner", guildId: "g1", channelId: "c1", conversationId: "conversation", objective: "Investigate the history" };
function open(filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`), session = "process-one") {
    const store = new TaskStore(filename, session);
    stores.push(store);
    return store;
}

describe("durable task records", () => {
    it("moves inactive owned work privately, preserving notes and rejecting competing or unresolved handoffs", async () => {
        const store = open();
        const taskId = await store.create(request);
        const workspace = await store.workspace(taskId, request.actorId, request.channelId, request.guildId);
        await workspace.addRequestNote({ requestId: "turn", threadId: "conversation", kind: "note", label: null, body: "Project checkpoint" });
        const move = { taskId, actorId: request.actorId, fromGuildId: request.guildId, fromChannelId: request.channelId, guildId: "g2", channelId: "c2", conversationId: "second" };
        await expect(store.handoff(move)).rejects.toThrow("inactive");
        await store.finish(taskId, request.actorId, "paused", "checkpoint");
        await expect(store.handoff({ ...move, actorId: "other" })).rejects.toThrow();
        const results = await Promise.allSettled([store.handoff(move), store.handoff({ ...move, channelId: "c3" })]);
        expect(results.filter(result => result.status === "fulfilled")).toHaveLength(1);
        const location = await store.ownedLocation(taskId, request.actorId);
        expect(location?.guildId).toBe("g2");
        expect(await store.privateOnly(taskId, request.actorId)).toBe(true);
        expect(await store.snapshot(taskId, request.actorId, request.channelId, request.guildId)).toBeNull();
        expect((await store.snapshot(taskId, request.actorId, location!.channelId, "g2"))?.notes[0].body).toBe("Project checkpoint");
        expect(await store.resume(taskId, request.actorId, location!.channelId, "g2")).toBe(request.objective);
        const action = await store.beginAction({ ...request, taskId, guildId: "g2", channelId: location!.channelId, invocationId: "write", tool: "send_message", arguments: {} });
        await store.settleAction(action, request.actorId, "unknown", null);
        await store.finish(taskId, request.actorId, "paused", "verify");
        await expect(store.handoff({ ...move, fromGuildId: "g2", fromChannelId: location!.channelId })).rejects.toThrow("resolved actions");
    });
    it("requires an owned paused task and explicit verification before resolving an unknown action", async () => {
        const store = open();
        const id = await store.create(request);
        const actionId = await store.beginAction({ ...request, taskId: id, invocationId: "uncertain", tool: "send_message", arguments: { channel_id: "c1" } });
        await store.settleAction(actionId, "owner", "unknown", null, "Connection lost");
        const resolution = { ...request, taskId: id, actionId, resolution: "applied" as const, verification: "Verified message m1 in channel c1" };
        await expect(store.resolveAction(resolution)).rejects.toThrow("paused task");
        await store.finish(id, "owner", "paused", "Verify the send");
        expect(await store.resume(id, "owner", "c1", "g1")).toBeNull();
        await expect(store.resolveAction({ ...resolution, actorId: "other" })).rejects.toThrow();
        await expect(store.resolveAction({ ...resolution, channelId: "c2" })).rejects.toThrow();
        await store.resolveAction(resolution);
        expect((await store.snapshot(id, "owner", "c1", "g1"))?.actions[0]).toMatchObject({ status: "succeeded", error: expect.stringContaining("owner_attestation") });
        await expect(store.resolveAction(resolution)).rejects.toThrow();
        expect(await store.resume(id, "owner", "c1", "g1")).toBe(request.objective);
    });
    it("retains known and unavailable usage across restart without exposing another task", async () => {
        const filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`);
        const store = open(filename);
        const id = await store.create(request);
        const usage = { ...request, taskId: id, requestId: "turn", model: "configured/model", promptTokens: 100, completionTokens: 20, durationMs: 150, estimatedCostUsd: 0.001 };
        await store.recordUsage(usage);
        await store.recordUsage({ ...usage, promptTokens: null, completionTokens: null, estimatedCostUsd: null });
        await expect(store.recordUsage({ ...usage, actorId: "other" })).rejects.toThrow("not owned");
        await store.close();
        const next = open(filename, "next");
        expect(await next.usage(id, "owner", "c1", "g1")).toHaveLength(2);
        expect((await next.usage(id, "owner", "c1", "g1")).some(row => row.promptTokens === null && row.estimatedCostUsd === null)).toBe(true);
        expect(await next.usage(id, "other", "c1", "g1")).toEqual([]);
        expect(await next.usage(id, "owner", "c2", "g1")).toEqual([]);
    });
    it("survives resetting disposable runtime and index data", async () => {
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        const store = open(path.join(AppPaths.storageRoot, "tasks.sqlite"));
        const id = await store.create(request);
        const { RuntimeStorageService } = await import("@/runtime/storage/RuntimeStorageService");
        await RuntimeStorageService.resetAllRuntimeData();
        expect(await store.ownsActive(id, "owner", "c1", "g1")).toBe(true);
    });
    it("isolates task visibility and continuation ownership by authenticated actor and location", async () => {
        const store = open();
        const id = await store.create(request);
        expect(await store.ownsActive(id, "owner", "c1", "g1")).toBe(true);
        expect(await store.ownsActive(id, "other", "c1", "g1")).toBe(false);
        expect(await store.list("other", "c1", "g1")).toEqual([]);
        expect(await store.list("owner", "c2", "g1")).toEqual([]);
        expect(await store.list("owner", "c1", "g2")).toEqual([]);
        expect(await store.list("owner", "c1", "g1")).toMatchObject([{ id, status: "running" }]);
    });

    it("does not permit another owner or a duplicate completion to overwrite an outcome", async () => {
        const store = open();
        const id = await store.create(request);
        expect(await store.finish(id, "other", "completed", "forged")).toBe(false);
        const results = await Promise.all([
            store.finish(id, "owner", "completed", "first"),
            store.finish(id, "owner", "failed", "second"),
        ]);
        expect(results.filter(Boolean)).toHaveLength(1);
        const [saved] = await store.list("owner", "c1", "g1");
        expect(saved.status).not.toBe("running");
        expect(await store.finish(id, "owner", "cancelled", "overwrite")).toBe(false);
        expect((await store.list("owner", "c1", "g1"))[0]).toEqual(saved);
    });

    it("preserves completed work and pauses interrupted work on a new process without replay", async () => {
        const filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`);
        const first = open(filename);
        const finished = await first.create(request);
        const receipt = await first.beginAction({ ...request, taskId: finished, invocationId: "sent", tool: "send_message", arguments: {} });
        await expect(first.settleAction(receipt, "other", "succeeded", {})).rejects.toThrow();
        await first.settleAction(receipt, "owner", "succeeded", { messageId: "m1" });
        await first.appendSteering(finished, "owner", "c1", "g1", "Keep source links");
        const decided = await first.beginApproval({ ...request, taskId: finished, request: { tool: "send_message" } });
        await first.settleApproval(decided, "owner", { approved: true, decidedBy: "owner" });
        await first.finish(finished, "owner", "completed", "The result");
        const interrupted = await first.create(request);
        const pending = await first.beginApproval({ ...request, taskId: interrupted, request: { tool: "delete_channel" } });
        await first.beginAction({ ...request, taskId: interrupted, invocationId: "pending", tool: "send_message", arguments: {} });
        await first.close();
        const next = open(filename, "process-two");
        const tasks = await next.list("owner", "c1", "g1");
        expect(tasks.find(task => task.id === finished)).toMatchObject({ status: "completed", answer: "The result" });
        expect(tasks.find(task => task.id === interrupted)).toMatchObject({ status: "paused", reason: "interrupted" });
        expect(await next.ownsActive(interrupted, "owner", "c1", "g1")).toBe(false);
        expect(await next.steering(finished, "owner", "c1", "g1")).toEqual(["Keep source links"]);
        expect(await next.approvals(finished, "owner", "c1", "g1")).toMatchObject([{ status: "decided" }]);
        expect(await next.approvals(interrupted, "owner", "c1", "g1")).toMatchObject([{ status: "interrupted", result: null }]);
        await expect(next.settleApproval(pending, "owner", { approved: true })).rejects.toThrow();
        expect((await next.snapshot(finished, "owner", "c1", "g1"))?.actions)
            .toMatchObject([{ status: "succeeded", result: { messageId: "m1" } }]);
        expect((await next.snapshot(interrupted, "owner", "c1", "g1"))?.actions)
            .toMatchObject([{ status: "unknown", error: "interrupted" }]);
        await next.close();
        const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
        const audit = createClient({ url: pathToFileURL(filename).toString() });
        try {
            const events = await audit.execute({ sql: "SELECT kind,detail FROM task_events WHERE task_id=? ORDER BY seq", args: [interrupted] });
            expect(events.rows.map(row => [row.kind, row.detail])).toEqual([["started", null], ["paused", "interrupted"]]);
        } finally { audit.close(); }
    });

    it("claims an explicit resume once and refuses unresolved mutations or a different owner", async () => {
        const store = open();
        const id = await store.create(request);
        await store.finish(id, "owner", "paused", "Need more evidence");
        expect(await store.resume(id, "other", "c1", "g1")).toBeNull();
        const claims = await Promise.all([store.resume(id, "owner", "c1", "g1"), store.resume(id, "owner", "c1", "g1")]);
        expect(claims.filter(value => value !== null)).toEqual([request.objective]);
        const action = await store.beginAction({ ...request, taskId: id, invocationId: "unknown", tool: "send_message", arguments: {} });
        await store.settleAction(action, "owner", "unknown", null);
        await store.finish(id, "owner", "paused", "Unknown send");
        expect(await store.resume(id, "owner", "c1", "g1")).toBeNull();
    });

    it("reopens the same process connection without interrupting its own work", async () => {
        const store = open();
        const id = await store.create(request);
        await store.close();
        expect(await store.ownsActive(id, "owner", "c1", "g1")).toBe(true);
    });

    it("preserves an invalid database instead of resetting it", async () => {
        const filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`);
        fs.writeFileSync(filename, "invalid database");
        const store = open(filename);
        await expect(store.initialize()).rejects.toThrow();
        expect(fs.readFileSync(filename, "utf8")).toBe("invalid database");
    });
});

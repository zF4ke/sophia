import { afterEach, describe, expect, it, vi } from "vitest";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { durableApproval } from "@/runtime/tasks/TaskApprovals";
import { ExecutionControl } from "@/runtime/ExecutionControl";

afterEach(() => vi.restoreAllMocks());
const actor = { actorId: "owner", channelId: "c1", guildId: "g1" };
async function setup() {
    const taskId = await taskStore.create({ ...actor, conversationId: "conversation", objective: "Change a channel" });
    return { ...actor, taskId };
}
describe("durable task decisions", () => {
    it("records prompt expiry separately from an explicit denial", async () => {
        const context = await setup();
        await durableApproval(context, async () => ({ approved: false, decidedBy: "timeout" }))!({ requesterId: "owner" });
        expect(await taskStore.approvals(context.taskId, "owner", "c1", "g1")).toMatchObject([{ status: "expired" }]);
    });
    it("preserves the owner's restrictive task mode and rejects foreign changes", async () => {
        const context = await setup();
        expect(await taskStore.approvalMode(context.taskId, "owner")).toBe("inherit");
        expect(await taskStore.approvalMode(context.taskId, "owner", "ask")).toBe("ask");
        await expect(taskStore.approvalMode(context.taskId, "other", "inherit")).rejects.toThrow("owned");
        expect(await taskStore.approvalMode(context.taskId, "owner")).toBe("ask");
        await taskStore.finish(context.taskId, "owner", "paused", "waiting");
        expect(await taskStore.approvalMode(context.taskId, "owner")).toBe("ask");
        await expect(taskStore.approvalMode(context.taskId, "owner", "inherit")).rejects.toThrow("active");
    });
    it("rebinds a reused approval transport to the current task ledger", async () => {
        const first = await setup();
        const second = await setup();
        const transport = vi.fn(async () => ({ approved: true }));
        const original = durableApproval(first, transport)!;
        expect(durableApproval(first, original)).toBe(original);
        await durableApproval(second, original)!({ requesterId: "owner" });
        expect(await taskStore.approvals(first.taskId, "owner", "c1", "g1")).toEqual([]);
        expect(await taskStore.approvals(second.taskId, "owner", "c1", "g1")).toMatchObject([{ status: "decided" }]);
        expect(transport).toHaveBeenCalledOnce();
    });
    it("saves exact requests before prompting and decisions before returning", async () => {
        const context = await setup();
        const request = { requesterId: "owner", toolArgs: { channel_id: "c1", name: "new" } };
        const decision = { approved: true, decidedBy: "owner", decidedAt: 123 };
        const gate = durableApproval(context, async () => {
            expect(await taskStore.approvals(context.taskId, "owner", "c1", "g1")).toMatchObject([{ status: "pending", request }]);
            return decision;
        })!;
        expect(await gate(request)).toEqual(decision);
        expect(await taskStore.approvals(context.taskId, "owner", "c1", "g1")).toMatchObject([{ status: "decided", result: decision }]);
        expect(await taskStore.approvals(context.taskId, "other", "c1", "g1")).toEqual([]);
        await expect(gate({ ...request, requesterId: "other" })).rejects.toThrow();
    });

    it("does not return permission if saving its decision fails", async () => {
        const context = await setup();
        vi.spyOn(taskStore, "settleApproval").mockRejectedValue(new Error("Disk unavailable"));
        const gate = durableApproval(context, async () => ({ approved: true }))!;
        await expect(gate({ requesterId: "owner" })).rejects.toMatchObject({ reason: "persistence_failed" });
    });

    it("saves steering before acknowledging it and stops on persistence failure", async () => {
        const context = await setup();
        const execution = new ExecutionControl("owner", "c1");
        execution.bindTask(context.taskId, text => taskStore.appendSteering(context.taskId, "owner", "c1", "g1", text));
        const release = execution.register();
        try {
            expect(await ExecutionControl.steerForActor("owner", "c1", "Keep the channel private")).toBe("queued");
            expect(await taskStore.steering(context.taskId, "owner", "c1", "g1")).toEqual(["Keep the channel private"]);
            vi.spyOn(taskStore, "appendSteering").mockRejectedValue(new Error("Disk unavailable"));
            await expect(ExecutionControl.steerForActor("owner", "c1", "Stop")).rejects.toMatchObject({ reason: "persistence_failed" });
            expect(() => execution.checkpoint()).toThrow();
        } finally { release(); }
    });
});

import { expect, it, vi } from "vitest";
import { createApprovalGate, createBatchApprovalGate, getPendingApproval, getPendingBatchApproval } from "@/discord/approval/ApprovalGate";
import { durableApproval } from "@/runtime/tasks/TaskApprovals";
import { taskStore } from "@/runtime/tasks/TaskStore";

it.each([false, true])("cancels an approval wait and removes its pending controls, batch=%s", async batch => {
    const controller = new AbortController();
    const context = { actorId: "owner", channelId: "channel", guildId: "guild", signal: controller.signal };
    const taskId = await taskStore.create({ ...context, conversationId: "conversation", objective: "Change a channel" });
    const edit = vi.fn().mockResolvedValue({});
    const transport = { send: vi.fn().mockResolvedValue({ edit }) };
    const item = { toolName: "create_channel" as const, toolArgs: { name: "test" }, description: "Create test", sideEffectLevel: "write" as const };
    const pending = batch
        ? durableApproval({ ...context, taskId }, createBatchApprovalGate(transport))!({ batchId: taskId, requesterId: "owner", items: [{ ...item, toolCallId: "call", targetCategory: null }] })
        : durableApproval({ ...context, taskId }, createApprovalGate(transport))!({ ...item, requestId: taskId, requesterId: "owner" });
    const rejected = expect(pending).rejects.toMatchObject({ reason: "cancelled" });
    await vi.waitFor(() => expect(batch ? getPendingBatchApproval(taskId) : getPendingApproval(taskId)).toBeDefined());
    controller.abort();
    await rejected;
    expect(batch ? getPendingBatchApproval(taskId) : getPendingApproval(taskId)).toBeUndefined();
    expect(edit).toHaveBeenCalledOnce();
    expect(await taskStore.approvals(taskId, "owner", "channel", "guild")).toMatchObject([{ status: "failed" }]);
    await taskStore.finish(taskId, "owner", "cancelled", "Stopped");
    expect(await taskStore.canResume(taskId, "owner", "channel", "guild")).toBe(true);
});

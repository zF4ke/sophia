import { expect, it, vi } from "vitest";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { ActionVerifier } from "@/runtime/tasks/ActionVerifier";

async function receipt(tool: string, args: Record<string, unknown>) {
    const owner = { actorId: "owner", guildId: "guild", channelId: "channel" };
    const taskId = await taskStore.create({ ...owner, conversationId: "channel", objective: "Change" });
    const actionId = await taskStore.beginAction({ ...owner, taskId, invocationId: "call", tool, arguments: args });
    await taskStore.settleAction(actionId, "owner", "unknown", null, "Connection lost");
    await taskStore.finish(taskId, "owner", "paused", "Verify result");
    return { taskId, actionId, snapshot: () => taskStore.snapshot(taskId, "owner", "channel", "guild") };
}
it("settles definite absence without replaying or resuming and rejects foreign owners", async () => {
    const saved = await receipt("delete_channel", { channel_id: "deleted" });
    const fetch = vi.fn().mockRejectedValue(Object.assign(new Error("Unknown channel"), { code: 10003 }));
    const context = { actorId: "owner", currentChannelId: "channel", guild: { id: "guild", channels: { fetch } } as never, question: "Verify", authorize: async () => "allow" as const };
    await expect(ActionVerifier.verify({ ...context, actorId: "other" }, saved.taskId, saved.actionId)).rejects.toThrow("not owned");
    expect(fetch).not.toHaveBeenCalled();
    expect(await ActionVerifier.verify(context, saved.taskId, saved.actionId)).toMatchObject({ resolution: "applied", resumed: false });
    expect((await saved.snapshot())?.actions[0]).toMatchObject({ status: "succeeded", error: expect.stringContaining("discord_observation") });
    expect(await taskStore.ownsActive(saved.taskId, "owner", "channel", "guild")).toBe(false);
});
it("does not mistake permission errors for an absent resource", async () => {
    const saved = await receipt("delete_channel", { channel_id: "private" });
    const context = { actorId: "owner", currentChannelId: "channel", guild: { id: "guild", channels: { fetch: vi.fn().mockRejectedValue(Object.assign(new Error("Missing access"), { code: 50001 })) } } as never, question: "Verify", authorize: async () => "allow" as const };
    await expect(ActionVerifier.verify(context, saved.taskId, saved.actionId)).rejects.toThrow("Missing access");
    expect((await saved.snapshot())?.actions[0].status).toBe("unknown");
});
it("keeps partial message deletion unresolved and does not infer sends from matching text", async () => {
    const saved = await receipt("delete_messages", { channel_id: "channel", message_ids: ["gone", "present"] });
    const messages = { fetch: vi.fn(async ({ message }: { message: string }) => message === "gone" ? null : { id: message }) };
    const channel = { isTextBased: () => true, permissionsFor: () => ({ has: () => true }), messages };
    const context = { actorId: "owner", currentChannelId: "channel", guild: { id: "guild", members: { fetch: vi.fn().mockResolvedValue({}) }, channels: { fetch: vi.fn().mockResolvedValue(channel) } } as never, question: "Verify", authorize: async () => "allow" as const };
    expect(await ActionVerifier.verify(context, saved.taskId, saved.actionId)).toMatchObject({ resolution: "unknown", detail: expect.stringContaining("partial") });
    expect((await saved.snapshot())?.actions[0].status).toBe("unknown");
    const send = await receipt("send_message", { channel_id: "channel", content: "Hello" });
    messages.fetch.mockClear();
    expect(await ActionVerifier.verify(context, send.taskId, send.actionId)).toMatchObject({ resolution: "unknown", detail: expect.stringContaining("No exact message ID") });
    expect(messages.fetch).not.toHaveBeenCalled();
});

import { afterEach, describe, expect, it, vi } from "vitest";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { ToolExecutor } from "@/runtime/ToolExecutor";
import { ExecutionControl } from "@/runtime/ExecutionControl";
import { taskStore } from "@/runtime/tasks/TaskStore";

afterEach(() => vi.restoreAllMocks());
async function setup() {
    const taskId = await taskStore.create({ actorId: "owner", guildId: "g1", channelId: "c1", conversationId: "thread", objective: "Send" });
    const context = { taskId, actorId: "owner", guild: { id: "g1" } as never, currentChannelId: "c1", question: "Send",
        authorize: vi.fn().mockResolvedValue("allow") };
    const original = CapabilityRegistry.get("send_message");
    const run = vi.fn().mockResolvedValue({ tool: "send_message", summary: "Sent m1 in c1", data: { messageId: "m1", channelId: "c1" } });
    vi.spyOn(CapabilityRegistry, "get").mockReturnValue({ ...original, run });
    return { taskId, context, run, args: { channel_id: "c1", content: "Hello" },
        snapshot: () => taskStore.snapshot(taskId, "owner", "c1", "g1") };
}

describe("durable action receipts", () => {
    it("records intent before dispatch, saves concrete results and prevents duplicate dispatch", async () => {
        const f = await setup();
        f.run.mockImplementationOnce(async () => {
            expect((await f.snapshot())?.actions).toMatchObject([{ status: "started", arguments: f.args }]);
            return { tool: "send_message", summary: "Sent m1", data: { messageId: "m1" } };
        });
        const result = await ToolExecutor.execute("send_message", f.args, { ...f.context, execution: new ExecutionControl("owner", "c1", 1) }, { invocationId: "call-1" });
        expect(result.record.blocked).not.toBe(true);
        expect((await f.snapshot())?.actions).toMatchObject([{ status: "succeeded", result: { data: { messageId: "m1" } } }]);
        const duplicate = await ToolExecutor.execute("send_message", f.args, f.context, { invocationId: "call-1" });
        expect(duplicate.record.blocked).toBe(true);
        expect(f.run).toHaveBeenCalledOnce();
    });

    it.each([new Error("Connection lost after sending"), null])("marks a dispatched failure unknown (%s)", async (error) => {
        const f = await setup();
        f.run.mockRejectedValue(error);
        const result = await ToolExecutor.execute("send_message", f.args, f.context);
        expect(result.uncertainAction).toBe(true);
        expect(result.record.summary).toContain("unknown");
        expect((await f.snapshot())?.actions).toMatchObject([{ status: "unknown", error: error instanceof Error ? error.message : String(error) }]);
    });

    it("does not dispatch if the durable intent cannot be written", async () => {
        const f = await setup();
        vi.spyOn(taskStore, "beginAction").mockRejectedValue(new Error("Disk unavailable"));
        const result = await ToolExecutor.execute("send_message", f.args, f.context);
        expect(result.record.blocked).toBe(true);
        expect(f.run).not.toHaveBeenCalled();
        expect((await f.snapshot())?.actions).toEqual([]);
    });

    it("records a skipped attempt when authority changes during intent persistence", async () => {
        const f = await setup();
        f.context.authorize.mockResolvedValueOnce("allow").mockResolvedValue("deny");
        const result = await ToolExecutor.execute("send_message", f.args, f.context);
        expect(result.uncertainAction).toBe(false);
        expect(f.run).not.toHaveBeenCalled();
        expect((await f.snapshot())?.actions).toMatchObject([{ status: "skipped" }]);
    });

    it("does not treat a lost success receipt as an ordinary retryable failure", async () => {
        const f = await setup();
        vi.spyOn(taskStore, "settleAction").mockRejectedValue(new Error("Disk unavailable"));
        const result = await ToolExecutor.execute("send_message", f.args, f.context);
        expect(f.run).toHaveBeenCalledOnce();
        expect(result.uncertainAction).toBe(true);
        expect((await f.snapshot())?.actions).toMatchObject([{ status: "started" }]);
    });
});

import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { ProgressStatusService } from "@/discord/responding/ProgressStatus";

function createMockChannel() {
    const sent: Array<{ content: string; id: string; edit: ReturnType<typeof vi.fn>; delete: ReturnType<typeof vi.fn> }> = [];
    let editThrows = false;
    let deleteThrows = false;
    let messageId = 0;

    return {
        sent,
        setEditThrows(v: boolean) { editThrows = v; },
        setDeleteThrows(v: boolean) { deleteThrows = v; },
        channel: {
            send: vi.fn(async (payload: { content: string }) => {
                const msg = {
                    id: `msg-${++messageId}`,
                    content: payload.content,
                    edit: vi.fn(async (next: { content: string }) => {
                        if (editThrows) throw Object.assign(new Error("Unknown message"), { code: 10008 });
                        msg.content = next.content;
                    }),
                    delete: vi.fn(async () => {
                        if (deleteThrows) throw new Error("Already deleted");
                    }),
                };
                sent.push(msg as any);
                return msg;
            }),
        } as any,
    };
}

describe("ProgressStatusService", () => {
    it("keeps private progress and cleanup on the interaction webhook", async () => {
        const send = vi.fn();
        const followUp = vi.fn().mockResolvedValue({ id: "private-progress" });
        const editMessage = vi.fn().mockResolvedValue(undefined);
        const deleteMessage = vi.fn().mockResolvedValue(undefined);
        const status = ProgressStatusService.startForInteraction({ channel: { send }, followUp, webhook: { editMessage, deleteMessage } } as never);
        status.bindTask("1234-abcd");
        await status.notify("Working");
        await status.finalize();
        expect(followUp).toHaveBeenCalledWith(expect.objectContaining({ flags: 64, allowedMentions: { parse: [] } }));
        expect(deleteMessage).toHaveBeenCalledWith("private-progress");
        expect(send).not.toHaveBeenCalled();
    });
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("sends a status message on first notify", async () => {
        const mock = createMockChannel();
        const ps = ProgressStatusService.startForChannel(mock.channel);

        await ps.notify("Searching…");

        expect(mock.channel.send).toHaveBeenCalledWith({ content: "-# Searching…", allowedMentions: { parse: [] } });
    });

    it("throttles rapid edits", async () => {
        const mock = createMockChannel();
        const ps = ProgressStatusService.startForChannel(mock.channel);

        await ps.notify("Step 1");
        await ps.notify("Step 2"); // Should be coalesced (within throttle window)

        // Only one send, no edits yet
        expect(mock.channel.send).toHaveBeenCalledTimes(1);
    });

    it("sends the latest coalesced summary after throttle window", async () => {
        const mock = createMockChannel();
        const ps = ProgressStatusService.startForChannel(mock.channel);

        await ps.notify("Step 1");
        await ps.notify("Step 2"); // Coalesced
        await ps.notify("Step 3"); // Coalesced, replaces Step 2

        // Advance past throttle
        await vi.advanceTimersByTimeAsync(1600);

        // The second call to send/edit should use the latest coalesced summary
        const sentMsg = mock.sent[0];
        expect(sentMsg.edit).toHaveBeenCalledWith({ content: "-# Step 3", allowedMentions: { parse: [] } });
    });

    it("finalize deletes the status message", async () => {
        const mock = createMockChannel();
        const ps = ProgressStatusService.startForChannel(mock.channel);

        await ps.notify("Working…");
        await ps.finalize();

        const sentMsg = mock.sent[0];
        expect(sentMsg.delete).toHaveBeenCalled();
    });

    it("finalize tolerates already-deleted messages", async () => {
        const mock = createMockChannel();
        mock.setDeleteThrows(true);
        const ps = ProgressStatusService.startForChannel(mock.channel);

        await ps.notify("Working…");
        // Should not throw
        await ps.finalize();
    });

    it("recreates message once if edit fails (user deleted it)", async () => {
        const mock = createMockChannel();
        const ps = ProgressStatusService.startForChannel(mock.channel);

        await ps.notify("Step 1");

        // Advance past throttle
        vi.advanceTimersByTime(1600);

        // Make the edit throw (simulating user-deleted message)
        mock.setEditThrows(true);

        await ps.notify("Step 2");

        // Should have called send twice: original + recreation
        expect(mock.channel.send).toHaveBeenCalledTimes(2);
    });

    it("renders controls for the bound task without triggering mentions", async () => {
        const mock = createMockChannel();
        const status = ProgressStatusService.startForChannel(mock.channel);
        status.bindTask("1234-abcd");
        await status.notify("Reading @everyone");
        const payload = mock.channel.send.mock.calls[0][0];
        const controls = payload.components[0].toJSON().components;
        expect(controls.map((button: { custom_id: string }) => button.custom_id)).toEqual(["task:stop:1234-abcd", "task:details:1234-abcd"]);
        expect(payload.allowedMentions).toEqual({ parse: [] });
        await status.finalize();
    });

    it("coalesces concurrent updates without duplicate messages or invented progress counts", async () => {
        const mock = createMockChannel();
        const ps = ProgressStatusService.startForChannel(mock.channel);

        await Promise.all([ps.notify("Searching @everyone"), ps.notify("Searching @everyone")]);

        await vi.advanceTimersByTimeAsync(1600);
        await ps.notify("Searching @everyone");

        const sentMsg = mock.sent[0];
        expect(mock.channel.send).toHaveBeenCalledOnce();
        expect(sentMsg.edit).not.toHaveBeenCalled();
        await ps.finalize();
        await ps.notify("Late result");
        expect(mock.channel.send).toHaveBeenCalledOnce();
    });
});

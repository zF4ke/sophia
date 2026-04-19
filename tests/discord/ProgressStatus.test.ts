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
            send: vi.fn(async (content: string) => {
                const msg = {
                    id: `msg-${++messageId}`,
                    content,
                    edit: vi.fn(async (newContent: string) => {
                        if (editThrows) throw new Error("Unknown message");
                        msg.content = newContent;
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

        expect(mock.channel.send).toHaveBeenCalledWith("-# ⏳ Searching…");
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
        vi.advanceTimersByTime(1600);

        await ps.notify("Step 3 (flush)");

        // The second call to send/edit should use the latest coalesced summary
        const sentMsg = mock.sent[0];
        expect(sentMsg.edit).toHaveBeenCalledWith("-# ⏳ Step 3");
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

    it("adds a repeat counter for consecutive identical summaries", async () => {
        const mock = createMockChannel();
        const ps = ProgressStatusService.startForChannel(mock.channel);

        await ps.notify("Running search_messages…");
        await ps.notify("Running search_messages…");

        vi.advanceTimersByTime(1600);
        await ps.notify("Running search_messages…");

        const sentMsg = mock.sent[0];
        expect(sentMsg.edit).toHaveBeenCalledWith("-# ⏳ Running search_messages… (x3)");
    });
});

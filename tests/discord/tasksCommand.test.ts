import { afterEach, describe, expect, it, vi } from "vitest";
import command from "@/discord/commands/tools/tasks.command";
import { taskStore } from "@/runtime/tasks/TaskStore";

afterEach(() => vi.restoreAllMocks());
describe("tasks command", () => {
    it("does not export a source-derived file after source access is revoked", async () => {
        vi.spyOn(taskStore, "snapshot").mockResolvedValue({ plan: null, goals: [], notes: [], actions: [] });
        vi.spyOn(taskStore, "files").mockResolvedValue([{ path: "private.txt", data: "YQ==", sourceChannelIds: ["source"], sourceMessageIds: ["m"] }]);
        const input = { user: { id: "owner" }, channelId: "channel", guildId: "guild",
            guild: { members: { fetch: vi.fn().mockResolvedValue({}) }, channels: { fetch: vi.fn().mockResolvedValue({ isTextBased: () => true, permissionsFor: () => ({ has: () => false }) }) } },
            options: { getBoolean: () => null, getString: (name: string) => name === "task_id" ? "task" : name === "file_path" ? "private.txt" : null },
            deferReply: vi.fn().mockResolvedValue({}), editReply: vi.fn().mockResolvedValue({}) };
        await command.execute(input as never);
        expect(input.editReply).toHaveBeenCalledWith({ content: "Já não tens acesso a uma das fontes deste ficheiro." });
    });
    it("uses authenticated identity and only renders the current location's records", async () => {
        const list = vi.spyOn(taskStore, "list").mockResolvedValue([{
            id: "task-id", objective: "Check @everyone <@123>", status: "paused", reason: "interrupted",
        }] as never);
        const input = { user: { id: "owner" }, channelId: "channel", guildId: "guild", options: { getBoolean: () => null, getString: () => null },
            deferReply: vi.fn().mockResolvedValue({}), editReply: vi.fn().mockResolvedValue({}) };
        await command.execute(input as never);
        expect(list).toHaveBeenCalledWith("owner", "channel", "guild");
        expect(input.deferReply).toHaveBeenCalledWith({ flags: undefined });
        expect(input.editReply).toHaveBeenCalledWith({ content: expect.stringContaining("Pausado após reinício"), allowedMentions: { parse: [] } });
    });

    it("exports only an owned task's working state in the channel by default", async () => {
        const snapshot = vi.spyOn(taskStore, "snapshot").mockResolvedValue({ plan: "Plan", goals: [], notes: [], actions: [] });
        const evidence = vi.spyOn(taskStore, "toolRuns").mockResolvedValue([]);
        const input = { user: { id: "owner" }, channelId: "channel", guildId: "guild", options: { getBoolean: () => null, getString: (name: string) => name === "task_id" ? "task-id" : null },
            deferReply: vi.fn().mockResolvedValue({}), editReply: vi.fn().mockResolvedValue({}) };
        await command.execute(input as never);
        expect(snapshot).toHaveBeenCalledWith("task-id", "owner", "channel", "guild");
        expect(evidence).toHaveBeenCalledWith("task-id", "owner", "channel", "guild");
        expect(input.deferReply).toHaveBeenCalledWith({ flags: undefined });
        expect(input.editReply).toHaveBeenCalledWith(expect.objectContaining({ files: [expect.objectContaining({ name: "task-notes.txt" })] }));
        evidence.mockResolvedValue([{ tool: "retrieve_messages", output: { data: { messageId: "m1" } } }] as never);
        snapshot.mockResolvedValue({ plan: null, goals: [], notes: [], actions: [] });
        input.editReply.mockClear();
        await command.execute(input as never);
        expect(input.editReply).toHaveBeenCalledWith(expect.objectContaining({ files: [expect.objectContaining({ name: "task-notes.txt" }),
            expect.objectContaining({ name: "task-evidence.json" })] }));
        snapshot.mockResolvedValue(null);
        input.editReply.mockClear();
        await command.execute(input as never);
        expect(input.editReply).toHaveBeenCalledWith({ content: "Não encontrei esse pedido entre os teus pedidos neste canal." });
    });
});

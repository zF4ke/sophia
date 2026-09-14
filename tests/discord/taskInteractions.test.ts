import { afterEach, expect, it, vi } from "vitest";
import { handleTaskInteraction } from "@/discord/responding/taskInteractions";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { ExecutionControl } from "@/runtime/ExecutionControl";
afterEach(() => vi.restoreAllMocks());

it("binds task controls to the authenticated owner and current location", async () => {
    const snapshot = vi.spyOn(taskStore, "snapshot").mockResolvedValue(null);
    const cancel = vi.spyOn(ExecutionControl, "cancelTask").mockReturnValue(true);
    const interaction = { customId: "task:stop:1234-abcd", user: { id: "owner" }, channelId: "channel", guildId: "guild", deferReply: vi.fn(), editReply: vi.fn() };
    await handleTaskInteraction(interaction as never);
    expect(snapshot).toHaveBeenCalledWith("1234-abcd", "owner", "channel", "guild");
    expect(cancel).not.toHaveBeenCalled();
    expect(interaction.deferReply).toHaveBeenCalledWith({ flags: 64 });
    snapshot.mockResolvedValue({ plan: null, goals: [], notes: [], actions: [] });
    await handleTaskInteraction(interaction as never);
    expect(cancel).toHaveBeenCalledWith("1234-abcd", "owner", "channel");
});

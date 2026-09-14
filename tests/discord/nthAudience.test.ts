import { afterEach, expect, it, vi } from "vitest";
import command from "@/discord/commands/tools/nth.command";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { taskStore } from "@/runtime/tasks/TaskStore";

afterEach(() => vi.restoreAllMocks());
it("keeps restricted indexed messages out of public cross-channel replies", async () => {
    const read = vi.spyOn(DiscordMemoryService, "getNthHistoricalMessageAsync").mockResolvedValue({ id: "message", content: "private source text", authorName: "Author", createdTimestamp: 1, jumpLink: "https://discord.com/channels/guild/source/message" } as never);
    const deleted = vi.spyOn(taskStore, "hasDeletedSources").mockResolvedValue(false);
    let ephemeral = false;
    const guild = { id: "guild", members: { fetch: vi.fn().mockResolvedValue({ id: "owner" }) }, channels: { fetch: vi.fn() } } as any;
    guild.channels.fetch.mockResolvedValue({ guild, isTextBased: () => true, permissionsFor: (principal: unknown) => ({ has: () => principal !== "guild" }) });
    const interaction = { guild, user: { id: "owner" }, channelId: "other", options: { getChannel: () => ({ id: "source" }), getInteger: () => 1, getBoolean: () => ephemeral }, deferReply: vi.fn().mockResolvedValue(undefined), editReply: vi.fn().mockResolvedValue(undefined) };
    await command.execute(interaction as never);
    expect(read).not.toHaveBeenCalled();
    ephemeral = true;
    await command.execute(interaction as never);
    expect(JSON.stringify(interaction.editReply.mock.calls.at(-1))).toContain("private source text");
    deleted.mockResolvedValue(true);
    await command.execute(interaction as never);
    expect(JSON.stringify(interaction.editReply.mock.calls.at(-1))).not.toContain("private source text");
});

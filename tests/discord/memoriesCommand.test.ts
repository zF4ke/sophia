import { afterEach, expect, it, vi } from "vitest";
import command from "@/discord/commands/tools/memories.command";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { SecurityService } from "@/security/SecurityService";

afterEach(() => vi.restoreAllMocks());
it("requires operator authority and an exact revision before adopting legacy memory", async () => {
    vi.spyOn(SecurityService, "initialize").mockResolvedValue();
    const admin = vi.spyOn(SecurityService, "isAdmin").mockReturnValue(false);
    const adopt = vi.spyOn(knowledgeStore, "adoptQuarantined").mockResolvedValue();
    let revision: number | null = 2;
    const interaction = { user: { id: "owner" }, guildId: "guild", channelId: "channel", options: {
        getBoolean: () => false, getString: (name: string) => name === "adopt" ? "memory-id" : null, getInteger: () => revision,
    }, deferReply: vi.fn().mockResolvedValue(undefined), editReply: vi.fn().mockResolvedValue(undefined) };
    await command.execute(interaction as never);
    expect(adopt).not.toHaveBeenCalled();
    admin.mockReturnValue(true);
    revision = null;
    await command.execute(interaction as never);
    expect(adopt).not.toHaveBeenCalled();
    revision = 2;
    await command.execute(interaction as never);
    expect(adopt).toHaveBeenCalledWith(expect.objectContaining({ actorId: "owner", guildId: "guild", privateResponse: false }), "memory-id", 2);
});

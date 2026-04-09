import { beforeEach, describe, expect, it, vi } from "vitest";
import { runWithRequestCacheContext } from "@/agent/orchestration/requestCacheContext";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";
import { MemoryDatabase } from "@/memory/MemoryDatabase";
import * as memberTools from "@/discord/tools/runtime/memberTools";

describe("DiscordToolService tool cache", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        vi.useFakeTimers();
        vi.setSystemTime(new Date("2026-04-09T06:00:00Z"));
        process.env.SOPHIA_MEMORY_DB_PATH = ":memory:";
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        MemoryDatabase.reset();
    });

    it("returns cached tool results on repeated calls", async () => {
        const getGuildContextSpy = vi
            .spyOn(memberTools, "getGuildContext")
            .mockResolvedValue({
                tool: "get_guild_context",
                summary: "Oz Synthesis: 17 membros e 68 canais.",
                data: {
                    id: "g1",
                    name: "Oz Synthesis",
                    memberCount: 17,
                    channelCount: 68,
                },
            } as any);

        const first = await runWithRequestCacheContext(
            { guildId: "g1", responseOrdinal: 1 },
            () =>
                DiscordToolService.getGuildContext({
                    id: "g1",
                    name: "Oz Synthesis",
                } as any)
        );
        const second = await runWithRequestCacheContext(
            { guildId: "g1", responseOrdinal: 2 },
            () =>
                DiscordToolService.getGuildContext({
                    id: "g1",
                    name: "Oz Synthesis",
                } as any)
        );

        expect(first.cacheStatus).toBe("miss");
        expect(second.cacheStatus).toBe("hit");
        expect(getGuildContextSpy).toHaveBeenCalledTimes(1);
    });

    it("expires cached tool results by ttl", async () => {
        const getGuildContextSpy = vi
            .spyOn(memberTools, "getGuildContext")
            .mockResolvedValue({
                tool: "get_guild_context",
                summary: "Oz Synthesis: 17 membros e 68 canais.",
                data: {
                    id: "g1",
                    name: "Oz Synthesis",
                    memberCount: 17,
                    channelCount: 68,
                },
            } as any);

        await runWithRequestCacheContext(
            { guildId: "g1", responseOrdinal: 1 },
            () => DiscordToolService.getGuildContext({ id: "g1", name: "Oz Synthesis" } as any)
        );
        vi.setSystemTime(new Date("2026-04-09T06:11:00Z"));
        const second = await runWithRequestCacheContext(
            { guildId: "g1", responseOrdinal: 2 },
            () =>
                DiscordToolService.getGuildContext({
                    id: "g1",
                    name: "Oz Synthesis",
                } as any)
        );

        expect(second.cacheStatus).toBe("miss");
        expect(getGuildContextSpy).toHaveBeenCalledTimes(2);
    });

    it("expires cached tool results after too many guild responses", async () => {
        const getGuildContextSpy = vi
            .spyOn(memberTools, "getGuildContext")
            .mockResolvedValue({
                tool: "get_guild_context",
                summary: "Oz Synthesis: 17 membros e 68 canais.",
                data: {
                    id: "g1",
                    name: "Oz Synthesis",
                    memberCount: 17,
                    channelCount: 68,
                },
            } as any);

        await runWithRequestCacheContext(
            { guildId: "g1", responseOrdinal: 1 },
            () => DiscordToolService.getGuildContext({ id: "g1", name: "Oz Synthesis" } as any)
        );

        const second = await runWithRequestCacheContext(
            { guildId: "g1", responseOrdinal: 8 },
            () => DiscordToolService.getGuildContext({ id: "g1", name: "Oz Synthesis" } as any)
        );

        expect(second.cacheStatus).toBe("miss");
        expect(getGuildContextSpy).toHaveBeenCalledTimes(2);
    });

    it("keeps distinct cache keys for different arguments", async () => {
        const getMemberProfileSpy = vi
            .spyOn(memberTools, "getMemberProfile")
            .mockResolvedValue({
                tool: "get_member_profile",
                summary: "scart (@scart) com 2 cargos visíveis.",
                data: {
                    id: "u1",
                    username: "scart",
                    displayName: "scart",
                    globalName: null,
                    nickname: null,
                    roles: ["music", "member"],
                    bannerUrl: null,
                    accentColor: null,
                    bio: null,
                },
            } as any);

        await runWithRequestCacheContext(
            { guildId: "g1", responseOrdinal: 1 },
            () => DiscordToolService.getMemberProfile({ id: "g1" } as any, "scart")
        );
        await runWithRequestCacheContext(
            { guildId: "g1", responseOrdinal: 2 },
            () => DiscordToolService.getMemberProfile({ id: "g1" } as any, "glonos")
        );

        expect(getMemberProfileSpy).toHaveBeenCalledTimes(2);
    });
});

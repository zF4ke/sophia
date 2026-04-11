import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DiscordChannelCrawlService } from "@/discord/live/DiscordChannelCrawlService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

describe("UnifiedMessageRetrieval", () => {
    beforeEach(async () => {
        vi.restoreAllMocks();
        process.env.RUNTIME_OPERATIONAL_DB_PATH = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `retrieval-${Date.now()}-${Math.random()}.sqlite`
        );
        process.env.RUNTIME_CHECKPOINT_DB_PATH = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `retrieval-checkpoint-${Date.now()}-${Math.random()}.sqlite`
        );
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        await DiscordMemoryService.resetForTests();
    });

    it("keeps live refresh scoped to resolved channel ids and uses crawl previews when the cache has no lexical hits", async () => {
        const searchSpy = vi
            .spyOn(DiscordMemoryService, "searchMessagesAsync")
            .mockResolvedValueOnce([])
            .mockResolvedValueOnce([]);
        const rankSpy = vi.spyOn(DiscordChannelCrawlService, "rankCandidateChannels");
        vi.spyOn(DiscordChannelCrawlService, "crawlChannelMessages").mockImplementation(
            async (_guild, channelId) =>
                ({
                    channelId,
                    channelName: channelId === "c-traveller" ? "traveller" : "atlas",
                    messagesFetched: 2,
                    messagesStored: 2,
                    exhausted: true,
                    queryHint: "me explique o que tem nesse traveller",
                    backgroundIngestQueued: true,
                    previewMessages: [
                        {
                            messageId: `${channelId}-m1`,
                            authorId: "u-bot",
                            authorName: "Service Bot",
                            content:
                                channelId === "c-traveller"
                                    ? "Traveller é um serviço de exploração e rotas."
                                    : "Atlas centraliza mapas e referências do servidor.",
                            createdTimestamp: 1700000000000,
                            jumpLink: `https://discord.com/channels/g1/${channelId}/${channelId}-m1`,
                        },
                    ],
                }) as any
        );
        vi.spyOn(DiscordChannelCrawlService, "waitForBackgroundIngest").mockResolvedValue(
            undefined
        );

        const result = await UnifiedMessageRetrieval.retrieve({
            guild: { id: "g1" } as any,
            question: "me explique o que tem nesse traveller",
            currentChannelId: "c-current",
            channelIds: ["c-traveller", "c-atlas"],
            limit: 4,
        });

        expect(rankSpy).not.toHaveBeenCalled();
        expect(searchSpy).toHaveBeenCalledWith(
            "me explique o que tem nesse traveller",
            expect.objectContaining({
                guildId: "g1",
                channelIds: ["c-traveller", "c-atlas"],
            }),
            4
        );
        expect(result.searchedChannelIds).toEqual(["c-traveller", "c-atlas"]);
        expect(result.fetchedChannelIds).toEqual(["c-traveller", "c-atlas"]);
        expect(result.liveEscalated).toBe(true);
        expect(result.cacheEnriched).toBe(true);
        expect(result.results).toEqual([
            expect.objectContaining({
                channelId: "c-traveller",
                channelName: "traveller",
                content: "Traveller é um serviço de exploração e rotas.",
            }),
            expect.objectContaining({
                channelId: "c-atlas",
                channelName: "atlas",
                content: "Atlas centraliza mapas e referências do servidor.",
            }),
        ]);
        expect(result.evidenceSufficient).toBe(true);
        expect(result.sourceOrigin).toBe("cache_after_refresh");
    });
});

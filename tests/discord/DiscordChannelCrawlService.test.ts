import { Collection, TextChannel } from "discord.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DiscordChannelCrawlService } from "@/discord/live/DiscordChannelCrawlService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { MemoryDatabase } from "@/memory/MemoryDatabase";

function createChannel(
    id: string,
    name: string,
    extra: Record<string, unknown> = {}
) {
    const channel = Object.create(TextChannel.prototype);
    Object.defineProperty(channel, "viewable", {
        value: true,
        configurable: true,
    });
    Object.assign(channel, {
        id,
        name,
        messages: {
            fetch: vi.fn(),
        },
        ...extra,
    });
    return channel;
}

describe("DiscordChannelCrawlService", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        process.env.SOPHIA_MEMORY_DB_PATH = ":memory:";
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        MemoryDatabase.reset();
    });

    it("ranks indexed and unindexed candidate channels together", () => {
        vi.spyOn(DiscordMemoryService, "getKnownChannels").mockReturnValue([
            {
                channelId: "c1",
                guildId: "g1",
                channelName: "scart",
                lastSeenTimestamp: 100,
            },
        ] as any);

        const guild = {
            id: "g1",
            channels: {
                cache: new Collection([
                    [
                        "c1",
                        createChannel("c1", "scart"),
                    ],
                    [
                        "c2",
                        createChannel("c2", "music-scart"),
                    ],
                ]),
            },
        } as any;

        const candidates = DiscordChannelCrawlService.rankCandidateChannels(
            guild,
            "musica do scart",
            null
        );

        expect(candidates).toEqual([
            expect.objectContaining({
                channelId: "c1",
                isIndexed: true,
            }),
            expect.objectContaining({
                channelId: "c2",
                isIndexed: false,
            }),
        ]);
    });

    it("crawls a channel, ingests messages, and records crawl state", async () => {
        const ingestMessage = vi
            .spyOn(DiscordMemoryService, "ingestMessage")
            .mockResolvedValue(undefined);
        const upsertChannel = vi
            .spyOn(DiscordMemoryService, "upsertDiscoveredChannel")
            .mockImplementation(() => undefined);
        const updateCrawlState = vi
            .spyOn(DiscordMemoryService, "updateChannelCrawlState")
            .mockImplementation(() => undefined);

        const fetchedMessages = new Collection([
            [
                "m2",
                { id: "m2", createdTimestamp: 200 } as any,
            ],
            [
                "m1",
                { id: "m1", createdTimestamp: 100 } as any,
            ],
        ]);
        const channel = createChannel("c1", "scart", {
            messages: {
                fetch: vi
                    .fn()
                    .mockResolvedValueOnce(fetchedMessages)
                    .mockResolvedValueOnce(new Collection()),
            },
        });
        const guild = {
            id: "g1",
            channels: {
                cache: new Collection([["c1", channel]]),
            },
        } as any;

        const result = await DiscordChannelCrawlService.crawlChannelMessages(
            guild,
            "c1",
            250,
            "scart"
        );
        await DiscordChannelCrawlService.waitForBackgroundIngest();

        expect(result.messagesFetched).toBe(2);
        expect(result.messagesStored).toBe(2);
        expect(result.exhausted).toBe(true);
        expect(result.backgroundIngestQueued).toBe(true);
        expect(result.previewMessages).toHaveLength(0);
        expect(ingestMessage).toHaveBeenCalledTimes(2);
        expect(upsertChannel).toHaveBeenCalledWith("c1", "g1", "scart", expect.any(Number));
        expect(updateCrawlState).toHaveBeenCalledWith("c1", "m1", true);
    });

    it("continues from the oldest fetched message instead of recrawling the top", async () => {
        DiscordMemoryService.updateChannelCrawlState("c1", "m-oldest", false);

        const fetchSpy = vi
            .fn()
            .mockResolvedValueOnce(new Collection())
            .mockResolvedValueOnce(new Collection());
        const channel = createChannel("c1", "silksong", {
            messages: {
                fetch: fetchSpy,
            },
        });
        const guild = {
            id: "g1",
            channels: {
                cache: new Collection([["c1", channel]]),
            },
        } as any;

        await DiscordChannelCrawlService.crawlChannelMessages(guild, "c1", 250, "silksong");

        expect(fetchSpy).toHaveBeenCalledWith({
            limit: 100,
            before: "m-oldest",
        });
    });
});

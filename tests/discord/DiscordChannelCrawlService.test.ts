import path from "path";
import { ChannelType, Collection } from "discord.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { DiscordChannelCrawlService } from "@/discord/live/DiscordChannelCrawlService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

function createChannel(
    id: string,
    name: string,
    extra: Record<string, unknown> = {}
) {
    return {
        id,
        name,
        type: ChannelType.GuildText,
        viewable: true,
        messages: {
            fetch: vi.fn(),
        },
        ...extra,
    };
}

describe("DiscordChannelCrawlService", () => {
    beforeEach(async () => {
        vi.restoreAllMocks();
        const operationalDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `crawl-${Date.now()}-${Math.random()}.sqlite`
        );
        const checkpointDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `crawl-checkpoint-${Date.now()}-${Math.random()}.sqlite`
        );
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath,
                checkpointDbPath,
            },
        });
        await DiscordMemoryService.resetForTests();
    });

    it("ranks indexed and unindexed candidate channels together", async () => {
        vi.spyOn(DiscordMemoryService, "getKnownChannelsAsync").mockResolvedValue([
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

        const candidates = await DiscordChannelCrawlService.rankCandidateChannels(
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
            .mockResolvedValue(undefined);
        const updateCrawlState = vi
            .spyOn(DiscordMemoryService, "updateChannelCrawlState")
            .mockResolvedValue(undefined);

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
        expect(upsertChannel).toHaveBeenCalledWith(
            "c1",
            "g1",
            "scart",
            expect.any(Number),
            expect.objectContaining({
                channelType: String(ChannelType.GuildText),
                parentCategoryId: null,
                parentCategoryName: null,
            })
        );
        expect(updateCrawlState).toHaveBeenCalledWith("c1", "m1", true);
    });

    it("continues from the oldest fetched message instead of recrawling the top", async () => {
        await DiscordMemoryService.updateChannelCrawlState("c1", "m-oldest", false);

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

    it("falls back to recent fetched messages for previews when lexical matching finds nothing", async () => {
        const fetchedMessages = new Collection([
            [
                "m2",
                {
                    id: "m2",
                    createdTimestamp: 200,
                    content: "Traveller centraliza exploração e rotas.",
                    author: { id: "u1", username: "bot" },
                    guildId: "g1",
                    channelId: "c1",
                } as any,
            ],
            [
                "m1",
                {
                    id: "m1",
                    createdTimestamp: 100,
                    content: "Atlas organiza mapas e referências.",
                    author: { id: "u2", username: "guide" },
                    guildId: "g1",
                    channelId: "c1",
                } as any,
            ],
        ]);
        const channel = createChannel("c1", "services", {
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
            "descrição dos serviços"
        );

        expect(result.previewMessages).toEqual([
            expect.objectContaining({
                messageId: "m2",
                content: "Traveller centraliza exploração e rotas.",
            }),
            expect.objectContaining({
                messageId: "m1",
                content: "Atlas organiza mapas e referências.",
            }),
        ]);
    });

    it("prefers guild display name and nickname over username in preview messages", async () => {
        const fetchedMessages = new Collection([
            [
                "m1",
                {
                    id: "m1",
                    createdTimestamp: 100,
                    content: "A qualidade audiovisual desse vídeo é bizarra.",
                    author: {
                        id: "u1",
                        username: "oneperson",
                        globalName: "One Person",
                    },
                    member: {
                        displayName: "openrosen",
                        nickname: "openrosen",
                    },
                    guildId: "g1",
                    channelId: "c1",
                } as any,
            ],
        ]);
        const channel = createChannel("c1", "comandos", {
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

        const result = await DiscordChannelCrawlService.crawlChannelMessages(guild, "c1", 250, "vídeo");

        expect(result.previewMessages).toEqual([
            expect.objectContaining({
                messageId: "m1",
                authorName: "openrosen",
                authorUsername: "oneperson",
                authorNickname: "openrosen",
            }),
        ]);
    });

    it("uses precise snowflake conversion for targeted timestamp crawls", async () => {
        const beforeTimestamp = Date.parse("2025-02-10T00:00:00Z");
        const fetchSpy = vi
            .fn()
            .mockResolvedValueOnce(new Collection())
            .mockResolvedValueOnce(new Collection());
        const channel = createChannel("c1", "comandos", {
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

        await DiscordChannelCrawlService.crawlChannelMessagesAtTime(
            guild,
            "c1",
            beforeTimestamp,
            100,
            "youtube"
        );

        const discordEpoch = BigInt(1420070400000);
        const discordSnowflakeIncrement = BigInt(4194304);
        const expectedBefore =
            ((BigInt(beforeTimestamp) - discordEpoch) * discordSnowflakeIncrement).toString();

        expect(fetchSpy).toHaveBeenCalledWith({
            limit: 100,
            before: expectedBefore,
        });
    });
});


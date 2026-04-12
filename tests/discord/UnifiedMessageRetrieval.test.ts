import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DiscordChannelCrawlService } from "@/discord/live/DiscordChannelCrawlService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { ModelGateway } from "@/ai/ModelGateway";

async function ingestMessage(options: {
    id: string;
    channelId: string;
    channelName: string;
    content: string;
    createdTimestamp: number;
}) {
    await DiscordMemoryService.ingestStoredMessage({
        id: options.id,
        guildId: "g1",
        channelId: options.channelId,
        channelName: options.channelName,
        authorId: "u1",
        authorName: "alice",
        content: options.content,
        attachmentsJson: "[]",
        referenceMessageId: null,
        createdTimestamp: options.createdTimestamp,
        jumpLink: `https://discord.com/channels/g1/${options.channelId}/${options.id}`,
        isBot: 0,
    });
}

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
        expect(searchSpy).not.toHaveBeenCalled();
        expect(result.searchedChannelIds).toEqual(["c-traveller", "c-atlas"]);
        expect(result.fetchedChannelIds).toEqual(["c-traveller", "c-atlas"]);
        expect(result.liveEscalated).toBe(true);
        expect(result.cacheEnriched).toBe(true);
        expect(result.historyMessages).toEqual([
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
        expect(result.semanticMatches).toEqual([]);
        expect(result.combinedResults).toEqual(result.historyMessages);
        expect(result.evidenceSufficient).toBe(true);
        expect(result.sourceOrigin).toBe("cache_after_refresh");
    });

    it("keeps guild display name and nickname after cache_after_refresh when fetched messages lack member", async () => {
        vi.spyOn(ModelGateway, "embedTexts").mockResolvedValue([[1, 0, 0]]);

        const fetchedMessages = new Map([
            [
                "m1",
                {
                    id: "m1",
                    createdTimestamp: 1770600000000,
                    content: "A qualidade audiovisual desse vídeo é bizarra.",
                    author: {
                        id: "u-openrosen",
                        username: "oneperson",
                        globalName: "One Person",
                        bot: false,
                    },
                    member: null,
                    guildId: "g1",
                    channelId: "c-comandos",
                    channel: {
                        id: "c-comandos",
                        name: "comandos",
                        isTextBased: () => true,
                    },
                    url: "https://discord.com/channels/g1/c-comandos/m1",
                    attachments: {
                        map: () => [],
                    },
                    reference: null,
                    guild: null,
                } as any,
            ],
        ]);

        const fetchMember = vi.fn().mockResolvedValue({
            id: "u-openrosen",
            displayName: "openrosen",
            nickname: "openrosen",
            user: {
                id: "u-openrosen",
                username: "oneperson",
                globalName: "One Person",
            },
        });

        const guild = {
            id: "g1",
            members: {
                cache: new Map(),
                fetch: fetchMember,
            },
            channels: {
                cache: new Map([
                    [
                        "c-comandos",
                        {
                            id: "c-comandos",
                            name: "comandos",
                            type: 0,
                            viewable: true,
                            messages: {
                                fetch: vi
                                    .fn()
                                    .mockResolvedValueOnce(fetchedMessages)
                                    .mockResolvedValueOnce(new Map()),
                            },
                        },
                    ],
                ]),
            },
        } as any;

        const result = await UnifiedMessageRetrieval.retrieve({
            guild,
            question: "9 de fevereiro o openrosen mandou um link",
            channelIds: ["c-comandos"],
            mode: "history",
            afterTimestamp: 1770595200000,
            beforeTimestamp: 1770681600000,
            limit: 4,
        });

        expect(fetchMember).toHaveBeenCalledWith("u-openrosen");
        expect(result.sourceOrigin).toBe("cache_after_refresh");
        expect(result.historyMessages).toEqual([
            expect.objectContaining({
                messageId: "m1",
                authorName: "openrosen",
                authorUsername: "oneperson",
                authorNickname: "openrosen",
            }),
        ]);
    });

    it("continues semantic-only retrieval with a stable cursor when history is empty from the start", async () => {
        vi.spyOn(ModelGateway, "embedTexts").mockResolvedValue([[1, 0, 0]]);

        await ingestMessage({
            id: "101",
            channelId: "c-riddles",
            channelName: "riddles",
            content:
                "riddle alpha ".repeat(12) +
                "this message is intentionally long enough to count as strong semantic evidence.",
            createdTimestamp: 100,
        });
        await ingestMessage({
            id: "102",
            channelId: "c-riddles",
            channelName: "riddles",
            content:
                "riddle beta ".repeat(12) +
                "this message is intentionally long enough to count as strong semantic evidence.",
            createdTimestamp: 200,
        });
        await ingestMessage({
            id: "103",
            channelId: "c-riddles",
            channelName: "riddles",
            content:
                "riddle gamma ".repeat(12) +
                "this message is intentionally long enough to count as strong semantic evidence.",
            createdTimestamp: 300,
        });

        const firstPage = await UnifiedMessageRetrieval.retrieve({
            guild: { id: "g1" } as any,
            question: "riddle",
            channelIds: ["c-riddles"],
            mode: "semantic",
            limit: 2,
        });

        expect(firstPage.historyMessages).toEqual([]);
        expect(firstPage.semanticMatches.map((row) => row.messageId)).toEqual(["103", "102"]);
        expect(firstPage.continuation.continuationAvailable).toBe(true);
        expect(firstPage.continuation.history.continuationAvailable).toBe(false);
        expect(firstPage.continuation.semantic.continuationAvailable).toBe(true);
        expect(firstPage.continuation.semantic.cursor).toEqual({
            lastScore: firstPage.semanticMatches[1]?.totalScore,
            lastCreatedTimestamp: 200,
            lastMessageId: "102",
        });

        await ingestMessage({
            id: "104",
            channelId: "c-riddles",
            channelName: "riddles",
            content:
                "riddle delta ".repeat(12) +
                "this message is intentionally long enough to count as strong semantic evidence.",
            createdTimestamp: 400,
        });

        const secondPage = await UnifiedMessageRetrieval.retrieve({
            guild: { id: "g1" } as any,
            question: "riddle",
            channelIds: ["c-riddles"],
            mode: "semantic",
            limit: 2,
            cursor: {
                semantic: firstPage.continuation.semantic.cursor,
            },
            excludedMessageIds: firstPage.semanticMatches.map((row) => row.messageId),
        });

        expect(secondPage.historyMessages).toEqual([]);
        expect(secondPage.semanticMatches.map((row) => row.messageId)).toEqual(["101"]);
        expect(secondPage.semanticMatches.map((row) => row.messageId)).not.toContain("104");
        expect(secondPage.continuation.semantic.continuationAvailable).toBe(false);
    });

    it("continues history retrieval across multiple pages without duplicates", async () => {
        vi.spyOn(ModelGateway, "embedTexts").mockResolvedValue([[1, 0, 0]]);

        for (const [id, createdTimestamp] of [
            ["101", 100],
            ["102", 200],
            ["103", 300],
            ["104", 400],
            ["105", 500],
        ] as const) {
            await ingestMessage({
                id,
                channelId: "c-atlas",
                channelName: "atlas",
                content: `atlas message ${id}`,
                createdTimestamp,
            });
        }

        const page1 = await UnifiedMessageRetrieval.retrieve({
            guild: { id: "g1" } as any,
            question: "atlas history",
            channelIds: ["c-atlas"],
            mode: "history",
            limit: 2,
        });
        const page2 = await UnifiedMessageRetrieval.retrieve({
            guild: { id: "g1" } as any,
            question: "atlas history",
            channelIds: ["c-atlas"],
            mode: "history",
            limit: 2,
            cursor: { history: page1.continuation.history.perChannelOldestMessageId },
            excludedMessageIds: page1.historyMessages.map((row) => row.messageId),
        });
        const page3 = await UnifiedMessageRetrieval.retrieve({
            guild: { id: "g1" } as any,
            question: "atlas history",
            channelIds: ["c-atlas"],
            mode: "history",
            limit: 2,
            cursor: { history: page2.continuation.history.perChannelOldestMessageId },
            excludedMessageIds: [
                ...page1.historyMessages.map((row) => row.messageId),
                ...page2.historyMessages.map((row) => row.messageId),
            ],
        });

        expect(page1.historyMessages.map((row) => row.messageId)).toEqual(["104", "105"]);
        expect(page2.historyMessages.map((row) => row.messageId)).toEqual(["102", "103"]);
        expect(page3.historyMessages.map((row) => row.messageId)).toEqual(["101"]);
        expect(
            new Set([
                ...page1.historyMessages.map((row) => row.messageId),
                ...page2.historyMessages.map((row) => row.messageId),
                ...page3.historyMessages.map((row) => row.messageId),
            ]).size
        ).toBe(5);
        expect(page3.continuation.history.continuationAvailable).toBe(false);
    });

    it("does not inject crawl previews for strict temporal lookups", async () => {
        vi.spyOn(DiscordChannelCrawlService, "crawlChannelMessages").mockResolvedValue({
            channelId: "c-comandos",
            channelName: "comandos",
            messagesFetched: 3,
            messagesStored: 3,
            oldestFetchedMessageId: "preview-1",
            exhausted: false,
            queryHint: "youtube",
            backgroundIngestQueued: true,
            previewMessages: [
                {
                    messageId: "preview-1",
                    authorId: "u-open",
                    authorName: "Openrosen",
                    content: "random recent message outside requested day",
                    createdTimestamp: Date.now(),
                    jumpLink: "https://discord.com/channels/g1/c-comandos/preview-1",
                },
            ],
        } as any);
        vi.spyOn(DiscordChannelCrawlService, "waitForBackgroundIngest").mockResolvedValue(
            undefined
        );

        const result = await UnifiedMessageRetrieval.retrieve({
            guild: { id: "g1" } as any,
            question: "link do youtube",
            channelIds: ["c-comandos"],
            mode: "history",
            afterTimestamp: Date.parse("2025-02-09T00:00:00Z"),
            beforeTimestamp: Date.parse("2025-02-10T00:00:00Z"),
            limit: 10,
        });

        expect(result.historyMessages).toEqual([]);
        expect(result.combinedResults).toEqual([]);
        expect(result.sourceOrigin).toBe("live_refresh");
        expect(result.continuation.history.continuationAvailable).toBe(true);
        expect(result.continuation.continuationAvailable).toBe(true);
        expect(result.continuation.history.perChannelOldestMessageId["c-comandos"]).toBeTruthy();
    });

    it("retries strict scoped retrieval without exclusions when continuation inputs hide all rows", async () => {
        const historySpy = vi
            .spyOn(DiscordMemoryService, "getChannelHistoryPageAsync")
            .mockResolvedValueOnce([])
            .mockResolvedValueOnce([
                {
                    id: "m-artist",
                    guildId: "g1",
                    channelId: "c-comandos",
                    channelName: "comandos",
                    authorId: "u-open",
                    authorName: "Openrosen",
                    content: "a foto parecia o m4rkim",
                    attachmentsJson: "[]",
                    referenceMessageId: null,
                    createdTimestamp: 1707510000000,
                    jumpLink: "https://discord.com/channels/g1/c-comandos/m-artist",
                    isBot: 0,
                },
            ] as any);

        const result = await UnifiedMessageRetrieval.retrieve({
            guild: null,
            question: "de quem era?",
            channelIds: ["c-comandos"],
            mode: "history",
            authorId: "u-open",
            afterTimestamp: Date.parse("2026-02-09T00:00:00Z"),
            beforeTimestamp: Date.parse("2026-02-10T00:00:00Z"),
            excludedMessageIds: ["m-artist"],
            limit: 5,
        });

        expect(historySpy).toHaveBeenCalledTimes(2);
        expect(result.historyMessages.map((row) => row.messageId)).toEqual(["m-artist"]);
        expect(result.retrievalDiagnostics).toEqual(
            expect.objectContaining({
                strictScopedQuery: true,
                continuationInputsApplied: true,
                scopedEmptyRetryAttempted: true,
                scopedEmptyRetryRecovered: true,
                retryStrategy: "without_excluded",
            })
        );
    });
});

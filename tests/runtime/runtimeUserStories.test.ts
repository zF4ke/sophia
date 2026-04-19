import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway, type ToolChatResult } from "@/ai/ModelGateway";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import type { TurnInput } from "@/runtime/contracts";

function createInput(overrides: Partial<TurnInput> = {}): TurnInput {
    return {
        question: "test",
        user: { id: "u-requester" } as any,
        requesterDisplayName: "Requester",
        guild: { id: "g1", name: "Oz Synthesis" } as any,
        currentChannelId: "c1",
        nativeThreadId: null,
        requestedWebMode: "off",
        trigger: "talk",
        replyContext: null,
        referencedMessage: null,
        conversation: {
            key: "g1:c1:channel",
            kind: "channel",
            trigger: "talk",
            replyAnchorMessageId: null,
            nativeThreadId: null,
        },
        ...overrides,
    };
}

let toolCallCounter = 0;

function makeToolCallResult(calls: Array<{ name: string; args: Record<string, unknown> }>): ToolChatResult {
    return {
        content: null,
        toolCalls: calls.map((c) => ({
            id: `tc-${++toolCallCounter}`,
            type: "function" as const,
            function: { name: c.name, arguments: JSON.stringify(c.args) },
        })),
        finishReason: "tool_calls",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

function makeFinishResult(answer: string): ToolChatResult {
    return {
        content: null,
        toolCalls: [{
            id: `tc-${++toolCallCounter}`,
            type: "function" as const,
            function: { name: "finish", arguments: JSON.stringify({ answer }) },
        }],
        finishReason: "tool_calls",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

describe("runtime user stories", () => {
    beforeEach(async () => {
        vi.restoreAllMocks();
        toolCallCounter = 0;
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";

        vi.spyOn(DiscordMemoryService, "getRecentRuntimeRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getKnownChannelsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "recordToolRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
        // Default stall classifier: not a stall
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({ stall: false });
    });

    it("answers a follow-up directly from reused prior retrieval evidence without rerunning tools", async () => {
        const followUpToolRun = {
            requestId: "req-previous",
            toolName: "retrieve_messages",
            argumentsJson: JSON.stringify({
                query: "9 de fevereiro openrosen youtube",
                channelIds: ["c-comandos"],
                authorId: "u-open",
            }),
            summary: "ordered history evidence; 2 history and 0 semantic result(s) from cached Discord history.",
            learned: "Openrosen mencionou que parecia M4rkim.",
            outputJson: JSON.stringify({
                tool: "retrieve_messages",
                summary: "ordered history evidence",
                data: {
                    mode: "history",
                    sourceOrigin: "cache",
                    targetAuthorId: "u-open",
                    targetChannelIds: ["c-comandos"],
                    historyMessages: [
                        {
                            messageId: "m1",
                            channelId: "c-comandos",
                            channelName: "comandos",
                            guildId: "g1",
                            authorId: "u-open",
                            authorName: "Openrosen",
                            content: "A foto parecia o M4rkim nessa thumb.",
                            createdTimestamp: 1707510000000,
                            jumpLink: "https://discord.com/channels/g1/c-comandos/m1",
                            lexicalScore: 4,
                            semanticScore: 0,
                            recencyScore: 0,
                            totalScore: 4,
                        },
                    ],
                    semanticMatches: [],
                    combinedResults: [
                        {
                            messageId: "m1",
                            channelId: "c-comandos",
                            channelName: "comandos",
                            guildId: "g1",
                            authorId: "u-open",
                            authorName: "Openrosen",
                            content: "A foto parecia o M4rkim nessa thumb.",
                            createdTimestamp: 1707510000000,
                            jumpLink: "https://discord.com/channels/g1/c-comandos/m1",
                            lexicalScore: 4,
                            semanticScore: 0,
                            recencyScore: 0,
                            totalScore: 4,
                        },
                    ],
                    continuation: {
                        history: {
                            perChannelOldestMessageId: { "c-comandos": "m1" },
                            continuationAvailable: true,
                        },
                        semantic: {
                            cursor: null,
                            continuationAvailable: false,
                        },
                        continuationAvailable: true,
                    },
                },
            }),
            createdTimestamp: Date.now() - 5_000,
        };

        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([
            followUpToolRun as any,
        ]);

        // Model sees prior evidence in the system prompt and answers directly via finish
        vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValue(
            makeFinishResult("Sim, ele comparou a foto com o M4rkim.")
        );

        const result = await Runtime.answer(
            createInput({
                question: "ele comentou com qual artista parecia?",
            })
        );

        expect(result.answer).toBe("Sim, ele comparou a foto com o M4rkim.");
        expect(result.toolRuns).toEqual([]);
    });

    it("drops stale scoped targets when a new question explicitly retargets a different channel group", async () => {
        const priorMemberRun = {
            requestId: "req-member",
            toolName: "resolve_member_identity",
            argumentsJson: JSON.stringify({ query: "nMarkov" }),
            summary: "resolved nMarkov",
            learned: "Resolved the author target.",
            outputJson: JSON.stringify({
                tool: "resolve_member_identity",
                summary: "resolved nMarkov",
                data: {
                    query: "nMarkov",
                    resolvedId: "569277281046888488",
                    displayName: "nMarkov",
                    username: "nMarkov",
                    globalName: null,
                    nickname: null,
                    isBot: true,
                    isCurrentGuildMember: true,
                    source: "live_exact",
                    confidence: "exact",
                    roles: [],
                },
            }),
            createdTimestamp: Date.now() - 10_000,
        };

        const priorRetrievalRun = {
            requestId: "req-retrieval",
            toolName: "retrieve_messages",
            argumentsJson: JSON.stringify({
                query: "markov",
                channelIds: ["731278507740495882"],
                authorId: "569277281046888488",
            }),
            summary: "ordered history evidence",
            learned: "Retrieved prior scoped discussion messages.",
            outputJson: JSON.stringify({
                tool: "retrieve_messages",
                summary: "ordered history evidence",
                data: {
                    mode: "history",
                    sourceOrigin: "cache_after_refresh",
                    targetAuthorId: "569277281046888488",
                    targetChannelIds: ["731278507740495882"],
                    searchedChannelIds: ["731278507740495882"],
                    historyMessages: [
                        {
                            messageId: "m-disc-1",
                            channelId: "731278507740495882",
                            channelName: "discussão",
                            guildId: "g1",
                            authorId: "569277281046888488",
                            authorName: "nMarkov",
                            authorUsername: "nMarkov",
                            content: "Eu não sou eu.",
                            createdTimestamp: 1707510000000,
                            jumpLink: "https://discord.com/channels/g1/731278507740495882/m-disc-1",
                            lexicalScore: 4,
                        },
                    ],
                    semanticMatches: [],
                    combinedResults: [
                        {
                            messageId: "m-disc-1",
                            channelId: "731278507740495882",
                            channelName: "discussão",
                            guildId: "g1",
                            authorId: "569277281046888488",
                            authorName: "nMarkov",
                            authorUsername: "nMarkov",
                            content: "Eu não sou eu.",
                            createdTimestamp: 1707510000000,
                            jumpLink: "https://discord.com/channels/g1/731278507740495882/m-disc-1",
                            lexicalScore: 4,
                        },
                    ],
                    continuation: {
                        history: {
                            perChannelOldestMessageId: { "731278507740495882": "m-disc-1" },
                            continuationAvailable: true,
                        },
                        semantic: {
                            cursor: null,
                            continuationAvailable: false,
                        },
                        continuationAvailable: true,
                    },
                },
            }),
            createdTimestamp: Date.now() - 9_000,
        };

        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([
            priorMemberRun as any,
            priorRetrievalRun as any,
        ]);

        const previewSpy = vi.fn().mockResolvedValue(undefined);

        // Model sees stale evidence but decides to answer directly about a new topic
        vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValue(
            makeFinishResult("Os canais de Serviços incluem bot-commands e automation.")
        );

        const result = await Runtime.answer(
            createInput({
                question:
                    "estou com pressa e precisava de um resumo do que tem nos canais do Serviços. quero uma descrição de cada serviço",
                trigger: "mention",
                debugSession: {
                    setClassifying: vi.fn().mockResolvedValue(undefined),
                    setClassification: vi.fn().mockResolvedValue(undefined),
                    setPlanning: vi.fn().mockResolvedValue(undefined),
                    setToolRunning: vi.fn().mockResolvedValue(undefined),
                    setToolResult: vi.fn().mockResolvedValue(undefined),
                    setEvidenceSummary: vi.fn().mockResolvedValue(undefined),
                    setGenerating: vi.fn().mockResolvedValue(undefined),
                    finishSuccess: vi.fn().mockResolvedValue(undefined),
                    finishError: vi.fn().mockResolvedValue(undefined),
                    setTraceEvent: previewSpy,
                },
            })
        );

        expect(result.answer.length).toBeGreaterThan(0);
        // Model answered directly via finish — no tools were called
        expect(result.toolRuns).toEqual([]);
    });

    it("answers requester identity naturally after resolving the requester exactly", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_member_identity", args: { queries: ["u-requester"] } }])
            )
            .mockResolvedValueOnce(
                makeFinishResult("Tu és o Requester aqui no servidor.")
            );
        vi.spyOn(DiscordLiveService, "resolveMemberIdentity").mockResolvedValue({
            query: "u-requester",
            resolvedId: "u-requester",
            displayName: "Requester",
            username: "requester",
            globalName: null,
            nickname: null,
            isBot: false,
            isCurrentGuildMember: true,
            source: "live_id",
            confidence: "exact",
            roles: ["Admin"],
        });

        const result = await Runtime.answer(
            createInput({
                question: "Quem sou eu?",
            })
        );

        expect(result.answer).toBe("Tu és o Requester aqui no servidor.");
        expect(result.toolRuns.map((run) => run.tool)).toEqual(["resolve_member_identity"]);
        expect(DiscordLiveService.resolveMemberIdentity).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" }),
            "u-requester"
        );
    });

    it("combines member resolution, channel resolution, and message retrieval for a grounded explanation", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_member_identity", args: { queries: ["One Person"] } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_channel_targets", args: { targets: ["reflexoes"] } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "list_guild_structure", args: { targetText: "reflexoes" } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "retrieve_messages", args: { query: "do que o One Person está falando?", authorId: "u-one", channelIds: ["c-reflexoes"] } }])
            )
            .mockResolvedValueOnce(
                makeFinishResult("O One Person estava falando sobre a imagem e dizendo que ela era mais simbólica do que literal.")
            );
        vi.spyOn(DiscordLiveService, "resolveMemberIdentity").mockResolvedValue({
            query: "One Person",
            resolvedId: "u-one",
            displayName: "One Person",
            username: "oneperson",
            globalName: null,
            nickname: "One Person",
            isBot: false,
            isCurrentGuildMember: true,
            source: "live_search",
            confidence: "high",
            roles: [],
        });
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargetsBatch").mockResolvedValue([{
            query: "reflexoes",
            resolvedIds: ["c-reflexoes"],
            entries: [
                {
                    id: "c-reflexoes",
                    guildId: "g1",
                    name: "reflexoes",
                    type: "0",
                    position: null,
                    parentCategoryId: "cat-1",
                    parentCategoryName: "Text",
                    isReadable: true,
                    isViewable: true,
                    isIndexed: true,
                    source: "live",
                    missingOrDeletedPossible: false,
                },
            ],
            exactIdMatch: false,
            confidence: "high",
        }]);
        vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
            {
                id: "cat-1",
                guildId: "g1",
                name: "Text",
                type: "4",
                position: null,
                parentCategoryId: null,
                parentCategoryName: null,
                isReadable: false,
                isViewable: true,
                isIndexed: false,
                source: "live",
                missingOrDeletedPossible: false,
            },
            {
                id: "c-reflexoes",
                guildId: "g1",
                name: "reflexoes",
                type: "0",
                position: null,
                parentCategoryId: "cat-1",
                parentCategoryName: "Text",
                isReadable: true,
                isViewable: true,
                isIndexed: true,
                source: "live",
                missingOrDeletedPossible: false,
            },
        ]);
        vi.spyOn(UnifiedMessageRetrieval, "retrieve").mockResolvedValue({
            query: "do que o One Person está falando?",
            mode: "mixed",
            historyMessages: [
                {
                    messageId: "m1",
                    channelId: "c-reflexoes",
                    channelName: "reflexoes",
                    guildId: "g1",
                    authorId: "u-one",
                    authorName: "One Person",
                    content: "A imagem era mais simbólica do que literal.",
                    createdTimestamp: 1700000000000,
                    jumpLink: "https://discord.com/channels/g1/c-reflexoes/m1",
                    lexicalScore: 3,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 3,
                },
            ],
            semanticMatches: [],
            combinedResults: [
                {
                    messageId: "m1",
                    channelId: "c-reflexoes",
                    channelName: "reflexoes",
                    guildId: "g1",
                    authorId: "u-one",
                    authorName: "One Person",
                    content: "A imagem era mais simbólica do que literal.",
                    createdTimestamp: 1700000000000,
                    jumpLink: "https://discord.com/channels/g1/c-reflexoes/m1",
                    lexicalScore: 3,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 3,
                },
            ],
            cacheHit: true,
            liveEscalated: false,
            searchedChannelIds: ["c-reflexoes"],
            fetchedChannelIds: [],
            cacheEnriched: false,
            evidenceSufficient: true,
            strongResultCount: 1,
            weakResultCount: 0,
            historyMessageCount: 1,
            semanticMatchCount: 0,
            sourceOrigin: "cache",
            targetAuthorId: "u-one",
            targetChannelIds: ["c-reflexoes"],
            continuation: {
                history: {
                    perChannelOldestMessageId: { "c-reflexoes": "m1" },
                    continuationAvailable: true,
                },
                semantic: {
                    cursor: null,
                    continuationAvailable: false,
                },
                perChannelOldestMessageId: { "c-reflexoes": "m1" },
                continuationAvailable: true,
            },
            exhaustion: {
                historyExhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: true,
                exhaustedChannelIds: [],
                exhausted: false,
            },
            accumulatedWindow: {
                beforeTimestamp: null,
                afterTimestamp: null,
            },
            accumulatedUniqueCount: 1,
            beforeTimestamp: null,
            afterTimestamp: null,
            excludedMessageIds: [],
        });

        const result = await Runtime.answer(
            createInput({
                question: "do que o One Person está falando em #reflexoes?",
            })
        );

        expect(result.answer).toBe(
            "O One Person estava falando sobre a imagem e dizendo que ela era mais simbólica do que literal."
        );
        expect(result.toolRuns.map((run) => run.tool)).toEqual([
            "resolve_member_identity",
            "resolve_channel_targets",
            "list_guild_structure",
            "retrieve_messages",
        ]);
        expect(UnifiedMessageRetrieval.retrieve).toHaveBeenCalledWith(
            expect.objectContaining({
                authorId: "u-one",
                channelIds: ["c-reflexoes"],
            })
        );
    });

    it("can combine channel target resolution with guild structure discovery to answer structure questions", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_channel_targets", args: { targets: ["123456789012345678"] } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "list_guild_structure", args: {} }])
            )
            .mockResolvedValueOnce(
                makeFinishResult("Esse id corresponde ao canal #ideas, dentro da categoria Projects.")
            );
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargetsBatch").mockResolvedValue([{
            query: "123456789012345678",
            resolvedIds: ["123456789012345678"],
            entries: [
                {
                    id: "123456789012345678",
                    guildId: "g1",
                    name: "ideas",
                    type: "0",
                    position: null,
                    parentCategoryId: "cat-projects",
                    parentCategoryName: "Projects",
                    isReadable: true,
                    isViewable: true,
                    isIndexed: true,
                    source: "live",
                    missingOrDeletedPossible: false,
                },
            ],
            exactIdMatch: true,
            confidence: "exact",
        }]);
        vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
            {
                id: "cat-projects",
                guildId: "g1",
                name: "Projects",
                type: "4",
                position: null,
                parentCategoryId: null,
                parentCategoryName: null,
                isReadable: false,
                isViewable: true,
                isIndexed: false,
                source: "live",
                missingOrDeletedPossible: false,
            },
            {
                id: "123456789012345678",
                guildId: "g1",
                name: "ideas",
                type: "0",
                position: null,
                parentCategoryId: "cat-projects",
                parentCategoryName: "Projects",
                isReadable: true,
                isViewable: true,
                isIndexed: true,
                source: "live",
                missingOrDeletedPossible: false,
            },
        ]);

        const result = await Runtime.answer(
            createInput({
                question: "what channel is 123456789012345678?",
            })
        );

        expect(result.answer).toBe(
            "Esse id corresponde ao canal #ideas, dentro da categoria Projects."
        );
        expect(result.toolRuns.map((run) => run.tool)).toEqual([
            "resolve_channel_targets",
            "list_guild_structure",
        ]);
    });

    it("inspects a matched category and then retrieves scoped messages before describing available services, even when one target channel is not indexed", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_channel_targets", args: { targets: ["serviços"] } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "list_guild_structure", args: { targetText: "serviços" } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "retrieve_messages", args: { query: "que serviços estão disponíveis nesse servidor?", channelIds: ["c-bot-commands", "c-automation"] } }])
            )
            .mockResolvedValueOnce(
                makeFinishResult("Na categoria Serviços, vocês têm pelo menos o #bot-commands para comandos e o #automation para automações e integrações.")
            );
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargetsBatch").mockResolvedValue([{
            query: "serviços",
            resolvedIds: ["c-bot-commands", "c-automation"],
            entries: [
                {
                    id: "cat-services",
                    guildId: "g1",
                    name: "Serviços",
                    type: "4",
                    position: null,
                    parentCategoryId: null,
                    parentCategoryName: null,
                    isReadable: false,
                    isViewable: true,
                    isIndexed: false,
                    source: "live",
                    missingOrDeletedPossible: false,
                },
            ],
            exactIdMatch: false,
            confidence: "high",
        }]);
        vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
            {
                id: "cat-services",
                guildId: "g1",
                name: "Serviços",
                type: "4",
                position: null,
                parentCategoryId: null,
                parentCategoryName: null,
                isReadable: false,
                isViewable: true,
                isIndexed: false,
                source: "live",
                missingOrDeletedPossible: false,
            },
            {
                id: "c-bot-commands",
                guildId: "g1",
                name: "bot-commands",
                type: "0",
                position: null,
                parentCategoryId: "cat-services",
                parentCategoryName: "Serviços",
                isReadable: true,
                isViewable: true,
                isIndexed: true,
                source: "live",
                missingOrDeletedPossible: false,
            },
            {
                id: "c-automation",
                guildId: "g1",
                name: "automation",
                type: "0",
                position: null,
                parentCategoryId: "cat-services",
                parentCategoryName: "Serviços",
                isReadable: true,
                isViewable: true,
                isIndexed: false,
                source: "live",
                missingOrDeletedPossible: false,
            },
        ]);
        vi.spyOn(UnifiedMessageRetrieval, "retrieve").mockResolvedValue({
            query: "que serviços estão disponíveis nesse servidor?",
            mode: "mixed",
            historyMessages: [
                {
                    messageId: "m1",
                    channelId: "c-bot-commands",
                    channelName: "bot-commands",
                    guildId: "g1",
                    authorId: "u-bot",
                    authorName: "Service Bot",
                    content: "Use este canal para comandos e utilidades do bot.",
                    createdTimestamp: 1700000000000,
                    jumpLink: "https://discord.com/channels/g1/c-bot-commands/m1",
                    lexicalScore: 4,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 4,
                },
                {
                    messageId: "m2",
                    channelId: "c-automation",
                    channelName: "automation",
                    guildId: "g1",
                    authorId: "u-bot",
                    authorName: "Automation Bot",
                    content: "Este canal centraliza automações e integrações.",
                    createdTimestamp: 1700000001000,
                    jumpLink: "https://discord.com/channels/g1/c-automation/m2",
                    lexicalScore: 4,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 4,
                },
            ],
            semanticMatches: [],
            combinedResults: [
                {
                    messageId: "m1",
                    channelId: "c-bot-commands",
                    channelName: "bot-commands",
                    guildId: "g1",
                    authorId: "u-bot",
                    authorName: "Service Bot",
                    content: "Use este canal para comandos e utilidades do bot.",
                    createdTimestamp: 1700000000000,
                    jumpLink: "https://discord.com/channels/g1/c-bot-commands/m1",
                    lexicalScore: 4,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 4,
                },
                {
                    messageId: "m2",
                    channelId: "c-automation",
                    channelName: "automation",
                    guildId: "g1",
                    authorId: "u-bot",
                    authorName: "Automation Bot",
                    content: "Este canal centraliza automações e integrações.",
                    createdTimestamp: 1700000001000,
                    jumpLink: "https://discord.com/channels/g1/c-automation/m2",
                    lexicalScore: 4,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 4,
                },
            ],
            cacheHit: false,
            liveEscalated: true,
            searchedChannelIds: ["c-bot-commands", "c-automation"],
            fetchedChannelIds: ["c-automation"],
            cacheEnriched: true,
            evidenceSufficient: true,
            strongResultCount: 2,
            weakResultCount: 0,
            historyMessageCount: 2,
            semanticMatchCount: 0,
            sourceOrigin: "live_refresh",
            targetAuthorId: null,
            targetChannelIds: ["c-bot-commands", "c-automation"],
            continuation: {
                history: {
                    perChannelOldestMessageId: {
                        "c-bot-commands": "m1",
                        "c-automation": "m2",
                    },
                    continuationAvailable: true,
                },
                semantic: {
                    cursor: null,
                    continuationAvailable: false,
                },
                perChannelOldestMessageId: {
                    "c-bot-commands": "m1",
                    "c-automation": "m2",
                },
                continuationAvailable: true,
            },
            exhaustion: {
                historyExhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: true,
                exhaustedChannelIds: [],
                exhausted: false,
            },
            accumulatedWindow: {
                beforeTimestamp: null,
                afterTimestamp: null,
            },
            accumulatedUniqueCount: 2,
            beforeTimestamp: null,
            afterTimestamp: null,
            excludedMessageIds: [],
        });

        const result = await Runtime.answer(
            createInput({
                question: "que serviços estão disponíveis nesse servidor?",
            })
        );

        expect(result.answer).toBe(
            "Na categoria Serviços, vocês têm pelo menos o #bot-commands para comandos e o #automation para automações e integrações."
        );
        expect(result.toolRuns.map((run) => run.tool)).toEqual([
            "resolve_channel_targets",
            "list_guild_structure",
            "retrieve_messages",
        ]);
        expect(UnifiedMessageRetrieval.retrieve).toHaveBeenCalledWith(
            expect.objectContaining({
                channelIds: ["c-bot-commands", "c-automation"],
            })
        );
        expect(result.toolRuns.find((run) => run.tool === "retrieve_messages")?.data).toEqual(
            expect.objectContaining({
                liveEscalated: true,
                fetchedChannelIds: ["c-automation"],
                cacheEnriched: true,
            })
        );
    });

});

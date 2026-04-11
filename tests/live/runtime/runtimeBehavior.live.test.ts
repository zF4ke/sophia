import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import { CheckpointStore } from "@/runtime/storage/CheckpointStore";
import type { TurnInput } from "@/runtime/contracts";

const LIVE_MODEL_TESTS_ENABLED =
    process.env.LIVE_MODEL_TESTS === "1" &&
    !!process.env.OPENROUTER_API_KEY &&
    process.env.OPENROUTER_API_KEY !== "test-key";

const describeLive = LIVE_MODEL_TESTS_ENABLED ? describe : describe.skip;

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

describeLive("live runtime behavior", () => {
    beforeEach(async () => {
        vi.restoreAllMocks();
        process.env.DISCORD_TOKEN ||= "test-token";
        process.env.RUNTIME_MAX_RESEARCH_PASSES = "4";
        process.env.RUNTIME_MAX_TOOL_CALLS = "6";
        process.env.RUNTIME_OPERATIONAL_DB_PATH = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `runtime-live-${Date.now()}-${Math.random()}.sqlite`
        );
        process.env.RUNTIME_CHECKPOINT_DB_PATH = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `runtime-live-checkpoint-${Date.now()}-${Math.random()}.sqlite`
        );
        await CheckpointStore.reset();
        (Runtime as any).graphPromise = null;
        (Runtime as any).requestContext = new Map();

        vi.spyOn(DiscordMemoryService, "getRecentRuntimeRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "recordToolRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
    });

    it(
        "answers an exact member id naturally with the real model",
        async () => {
            vi.spyOn(DiscordLiveService, "resolveMemberIdentity").mockResolvedValue({
                query: "111111111111111111",
                resolvedId: "111111111111111111",
                displayName: "Markov",
                username: "markov_bot",
                globalName: null,
                nickname: null,
                isBot: true,
                isCurrentGuildMember: true,
                source: "live_id",
                confidence: "exact",
                roles: ["Bots"],
            });
            vi.spyOn(DiscordLiveService, "getMemberProfile").mockResolvedValue({
                id: "111111111111111111",
                query: "111111111111111111",
                username: "markov_bot",
                displayName: "Markov",
                globalName: null,
                nickname: null,
                roles: ["Bots"],
                bannerUrl: null,
                accentColor: null,
                bio: null,
                isBot: true,
                isCurrentGuildMember: true,
                source: "live_id",
                confidence: "exact",
            });

            const result = await Runtime.answer(
                createInput({
                    question: "Quem é <@111111111111111111>?",
                    trigger: "mention",
                })
            );

            const normalized = result.answer.toLowerCase();
            expect(
                result.toolRuns.some(
                    (run) =>
                        run.tool === "resolve_member_identity" || run.tool === "get_member_profile"
                )
            ).toBe(true);
            expect(normalized).toContain("markov");
            expect(normalized).not.toContain("based on what i found");
            expect(normalized).not.toContain("resolve_member_identity");
            expect(normalized).not.toContain("roles=");
        },
        120000
    );

    it(
        "uses grounded retrieval and still answers naturally with the real model",
        async () => {
            vi.spyOn(DiscordLiveService, "resolveMemberIdentity").mockResolvedValue({
                query: "111111111111111111",
                resolvedId: "111111111111111111",
                displayName: "One Person",
                username: "oneperson",
                globalName: null,
                nickname: "One Person",
                isBot: false,
                isCurrentGuildMember: true,
                source: "live_id",
                confidence: "exact",
                roles: [],
            });
            vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargets").mockResolvedValue({
                query: "123456789012345678",
                resolvedIds: ["123456789012345678"],
                entries: [
                    {
                        id: "123456789012345678",
                        guildId: "g1",
                        name: "reflexoes",
                        type: "0",
                        parentCategoryId: "cat-1",
                        parentCategoryName: "Text",
                        isReadable: true,
                        isViewable: true,
                        isIndexed: true,
                        source: "live",
                        missingOrDeletedPossible: false,
                    },
                ],
                exactIdMatch: true,
                confidence: "exact",
            });
            vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
                {
                    id: "cat-1",
                    guildId: "g1",
                    name: "Text",
                    type: "4",
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
                    name: "reflexoes",
                    type: "0",
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
                query: "O que <@111111111111111111> disse em <#123456789012345678> sobre a imagem?",
                mode: "mixed",
                historyMessages: [
                    {
                        messageId: "m1",
                        channelId: "123456789012345678",
                        channelName: "reflexoes",
                        guildId: "g1",
                        authorId: "111111111111111111",
                        authorName: "One Person",
                        content: "A imagem era mais simbólica do que literal.",
                        createdTimestamp: 1700000000000,
                        jumpLink: "https://discord.com/channels/g1/123456789012345678/m1",
                        lexicalScore: 5,
                        semanticScore: 0,
                        recencyScore: 0,
                        totalScore: 5,
                    },
                ],
                semanticMatches: [],
                combinedResults: [
                    {
                        messageId: "m1",
                        channelId: "123456789012345678",
                        channelName: "reflexoes",
                        guildId: "g1",
                        authorId: "111111111111111111",
                        authorName: "One Person",
                        content: "A imagem era mais simbólica do que literal.",
                        createdTimestamp: 1700000000000,
                        jumpLink: "https://discord.com/channels/g1/123456789012345678/m1",
                        lexicalScore: 5,
                        semanticScore: 0,
                        recencyScore: 0,
                        totalScore: 5,
                    },
                ],
                cacheHit: true,
                liveEscalated: false,
                searchedChannelIds: ["123456789012345678"],
                fetchedChannelIds: [],
                cacheEnriched: false,
                evidenceSufficient: true,
                strongResultCount: 1,
                weakResultCount: 0,
                historyMessageCount: 1,
                semanticMatchCount: 0,
                sourceOrigin: "cache",
                targetAuthorId: "111111111111111111",
                targetChannelIds: ["123456789012345678"],
                continuation: {
                    history: {
                        perChannelOldestMessageId: { "123456789012345678": "m1" },
                        continuationAvailable: true,
                    },
                    semantic: {
                        cursor: null,
                        continuationAvailable: false,
                    },
                    perChannelOldestMessageId: { "123456789012345678": "m1" },
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
                    question:
                        "O que <@111111111111111111> disse em <#123456789012345678> sobre a imagem?",
                    trigger: "mention",
                })
            );

            const tools = result.toolRuns.map((run) => run.tool);
            const normalized = result.answer.toLowerCase();

            expect(tools).toContain("retrieve_messages");
            expect(normalized).toContain("imagem");
            expect(normalized).not.toContain("based on what i found");
            expect(normalized).not.toContain("cached discord history");
            expect(normalized).not.toContain("retrieve_messages");
        },
        120000
    );

    it(
        "uses category discovery plus scoped retrieval for service-category questions with the real model",
        async () => {
            vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargets").mockResolvedValue({
                query: "serviços",
                resolvedIds: ["c-bot-commands", "c-automation"],
                entries: [
                    {
                        id: "cat-services",
                        guildId: "g1",
                        name: "Serviços",
                        type: "4",
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
            });
            vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
                {
                    id: "cat-services",
                    guildId: "g1",
                    name: "Serviços",
                    type: "4",
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
                    parentCategoryId: "cat-services",
                    parentCategoryName: "Serviços",
                    isReadable: true,
                    isViewable: true,
                    isIndexed: true,
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
                cacheHit: true,
                liveEscalated: false,
                searchedChannelIds: ["c-bot-commands", "c-automation"],
                fetchedChannelIds: [],
                cacheEnriched: false,
                evidenceSufficient: true,
                strongResultCount: 2,
                weakResultCount: 0,
                historyMessageCount: 2,
                semanticMatchCount: 0,
                sourceOrigin: "cache",
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
                    trigger: "reply",
                })
            );

            const tools = result.toolRuns.map((run) => run.tool);
            const normalized = result.answer.toLowerCase();

            expect(tools).toContain("resolve_channel_targets");
            expect(tools).toContain("list_guild_structure");
            expect(tools).toContain("retrieve_messages");
            expect(normalized).toContain("serv");
            expect(normalized).toContain("bot");
            expect(normalized).not.toContain("vazia");
            expect(normalized).not.toContain("empty");
        },
        120000
    );
});

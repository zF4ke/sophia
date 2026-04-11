import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway } from "@/ai/ModelGateway";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import { CheckpointStore } from "@/runtime/storage/CheckpointStore";
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

describe("runtime user stories", () => {
    beforeEach(async () => {
        vi.restoreAllMocks();
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        process.env.RUNTIME_MAX_RESEARCH_PASSES = "4";
        process.env.RUNTIME_MAX_TOOL_CALLS = "6";
        process.env.RUNTIME_OPERATIONAL_DB_PATH = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `runtime-story-${Date.now()}-${Math.random()}.sqlite`
        );
        process.env.RUNTIME_CHECKPOINT_DB_PATH = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `runtime-story-checkpoint-${Date.now()}-${Math.random()}.sqlite`
        );
        await CheckpointStore.reset();
        (Runtime as any).graphPromise = null;
        (Runtime as any).requestContext = new Map();

        vi.spyOn(DiscordMemoryService, "getRecentRuntimeRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "recordToolRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
    });

    it("answers requester identity naturally after resolving the requester exactly", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockImplementation(async (_messages, fallback, options) => {
            const traceLabel = options?.traceContext?.traceLabel;
            if (traceLabel === "runtime_plan_turn") {
                return {
                    mode: "research",
                    reason: "Resolve the requester exactly.",
                    goal: "Answer who the requester is.",
                    successCriteria: "Resolve the requester and answer naturally.",
                    candidateCapabilities: ["resolve_member_identity"],
                    confidence: "best_effort",
                } as any;
            }
            if (traceLabel === "runtime_judge_evidence") {
                const prompt = String((_messages?.[1] as any)?.content || "");
                if (prompt.includes("resolve_member_identity")) {
                    return {
                        sufficient: true,
                        confidence: "confident",
                        reason: "The requester identity is grounded.",
                    } as any;
                }
                return fallback as any;
            }
            if (traceLabel === "runtime_select_next_step") {
                return {
                    nextCapability: "resolve_member_identity",
                    arguments: { query: "u-requester" },
                    reason: "Resolve the requester directly.",
                    learnedExpectation: "Return the requester identity.",
                } as any;
            }
            return fallback as any;
        });
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "Tu és o Requester aqui no servidor."
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
        const selectStepCalls: string[] = [];

        vi.spyOn(ModelGateway, "generateJson").mockImplementation(async (_messages, fallback, options) => {
            const traceLabel = options?.traceContext?.traceLabel;
            if (traceLabel === "runtime_plan_turn") {
                return {
                    mode: "research",
                    reason: "Need to resolve the speaker, scope, and then retrieve messages.",
                    goal: "Explain what One Person was saying in #reflexoes.",
                    successCriteria: "Use exact scoped Discord evidence and answer naturally.",
                    candidateCapabilities: [
                        "resolve_member_identity",
                        "resolve_channel_targets",
                        "retrieve_messages",
                    ],
                    confidence: "best_effort",
                } as any;
            }
            if (traceLabel === "runtime_select_next_step") {
                const next = selectStepCalls.length === 0
                    ? "resolve_member_identity"
                    : selectStepCalls.length === 1
                      ? "resolve_channel_targets"
                      : selectStepCalls.length === 2
                        ? "list_guild_structure"
                        : "retrieve_messages";
                selectStepCalls.push(next);
                if (next === "resolve_member_identity") {
                    return {
                        nextCapability: next,
                        arguments: { query: "One Person" },
                        reason: "Resolve the speaker first.",
                        learnedExpectation: "Identify the target member.",
                    } as any;
                }
                if (next === "resolve_channel_targets") {
                    return {
                        nextCapability: next,
                        arguments: { targetText: "reflexoes" },
                        reason: "Resolve the target channel.",
                        learnedExpectation: "Identify the target channel ids.",
                    } as any;
                }
                if (next === "list_guild_structure") {
                    return {
                        nextCapability: next,
                        arguments: { targetText: "reflexoes" },
                        reason: "Inspect the matched channel structure before scoped retrieval.",
                        learnedExpectation: "Confirm the matched channel and its visibility before reading messages.",
                    } as any;
                }
                return {
                    nextCapability: next,
                    arguments: { query: "do que o One Person está falando?" },
                    reason: "Retrieve scoped message evidence.",
                    learnedExpectation: "Return the relevant messages.",
                } as any;
            }
            if (traceLabel === "runtime_judge_evidence") {
                const prompt = String((_messages?.[1] as any)?.content || "");
                if (prompt.includes("One Person estava falando sobre") || prompt.includes("retrieve_messages")) {
                    return {
                        sufficient: true,
                        confidence: "confident",
                        reason: "Scoped message evidence is available.",
                    } as any;
                }
                return {
                    sufficient: false,
                    confidence: "best_effort",
                    reason: "Need more current-guild evidence first.",
                } as any;
            }
            return fallback as any;
        });
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "O One Person estava falando sobre a imagem e dizendo que ela era mais simbólica do que literal."
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
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargets").mockResolvedValue({
            query: "reflexoes",
            resolvedIds: ["c-reflexoes"],
            entries: [
                {
                    id: "c-reflexoes",
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
            exactIdMatch: false,
            confidence: "high",
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
                id: "c-reflexoes",
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
        let selectCall = 0;

        vi.spyOn(ModelGateway, "generateJson").mockImplementation(async (_messages, fallback, options) => {
            const traceLabel = options?.traceContext?.traceLabel;
            if (traceLabel === "runtime_plan_turn") {
                return {
                    mode: "research",
                    reason: "Resolve the target and inspect the guild structure.",
                    goal: "Explain what channel/category an id belongs to.",
                    successCriteria: "Use guild discovery tools and answer naturally.",
                    candidateCapabilities: [
                        "resolve_channel_targets",
                        "list_guild_structure",
                    ],
                    confidence: "best_effort",
                } as any;
            }
            if (traceLabel === "runtime_select_next_step") {
                selectCall += 1;
                return selectCall === 1
                    ? {
                          nextCapability: "resolve_channel_targets",
                          arguments: { targetText: "123456789012345678" },
                          reason: "Resolve the explicit id first.",
                          learnedExpectation: "Find the exact channel target.",
                      }
                    : {
                          nextCapability: "list_guild_structure",
                          arguments: {},
                          reason: "Load the current guild structure.",
                          learnedExpectation: "Confirm category and visibility.",
                      };
            }
            if (traceLabel === "runtime_judge_evidence") {
                const prompt = String((_messages?.[1] as any)?.content || "");
                if (prompt.includes("ideas") && prompt.includes("Projects")) {
                    return {
                        sufficient: true,
                        confidence: "confident",
                        reason: "Guild structure evidence is available.",
                    } as any;
                }
                return {
                    sufficient: false,
                    confidence: "best_effort",
                    reason: "Need target resolution and guild structure first.",
                } as any;
            }
            return fallback as any;
        });
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "Esse id corresponde ao canal #ideas, dentro da categoria Projects."
        );
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargets").mockResolvedValue({
            query: "123456789012345678",
            resolvedIds: ["123456789012345678"],
            entries: [
                {
                    id: "123456789012345678",
                    guildId: "g1",
                    name: "ideas",
                    type: "0",
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
        });
        vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
            {
                id: "cat-projects",
                guildId: "g1",
                name: "Projects",
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
                name: "ideas",
                type: "0",
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
        let selectCall = 0;

        vi.spyOn(ModelGateway, "generateJson").mockImplementation(async (_messages, fallback, options) => {
            const traceLabel = options?.traceContext?.traceLabel;
            if (traceLabel === "runtime_plan_turn") {
                return {
                    mode: "research",
                    reason: "Resolve the service category, inspect it, and then retrieve scoped evidence.",
                    goal: "Describe what services are available in the matched service area.",
                    successCriteria: "Use structure plus scoped messages and answer naturally.",
                    candidateCapabilities: [
                        "resolve_channel_targets",
                        "list_guild_structure",
                        "retrieve_messages",
                    ],
                    confidence: "best_effort",
                } as any;
            }
            if (traceLabel === "runtime_select_next_step") {
                selectCall += 1;
                return selectCall === 1
                    ? {
                          nextCapability: "resolve_channel_targets",
                          arguments: { targetText: "serviços" },
                          reason: "Resolve the service category first.",
                          learnedExpectation: "Identify the service category and its child channels.",
                      }
                    : selectCall === 2
                      ? {
                            nextCapability: "list_guild_structure",
                            arguments: { targetText: "serviços" },
                            reason: "Inspect the matched category structure.",
                            learnedExpectation: "Confirm the visible channels inside Serviços.",
                        }
                      : {
                            nextCapability: "retrieve_messages",
                            arguments: { query: "que serviços estão disponíveis nesse servidor?" },
                            reason: "Read scoped message evidence from the resolved service channels.",
                            learnedExpectation: "Return messages that explain what the services do.",
                        };
            }
            if (traceLabel === "runtime_judge_evidence") {
                const prompt = String((_messages?.[1] as any)?.content || "");
                if (prompt.includes("#bot-commands") && prompt.includes("#automation")) {
                    return {
                        sufficient: true,
                        confidence: "confident",
                        reason: "The service category and scoped message evidence are available.",
                    } as any;
                }
                return {
                    sufficient: false,
                    confidence: "best_effort",
                    reason: "Need the service category plus scoped messages first.",
                } as any;
            }
            return fallback as any;
        });
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "Na categoria Serviços, vocês têm pelo menos o #bot-commands para comandos e o #automation para automações e integrações."
        );
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

import { beforeEach, describe, expect, it, vi } from "vitest";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";

describe("CapabilityRegistry", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
    });

    it("executes retrieve_messages through the unified retrieval pipeline", async () => {
        vi.spyOn(UnifiedMessageRetrieval, "retrieve").mockResolvedValue({
            query: "what did alice say in reflexoes",
            mode: "mixed",
            historyMessages: [
                {
                    messageId: "m1",
                    channelId: "c-reflexoes",
                    channelName: "reflexoes",
                    guildId: "g1",
                    authorId: "u-alice",
                    authorName: "Alice",
                    content: "The image was more symbolic than literal.",
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
                    authorId: "u-alice",
                    authorName: "Alice",
                    content: "The image was more symbolic than literal.",
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
            targetAuthorId: "u-alice",
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

        const capability = CapabilityRegistry.get("retrieve_messages");
        const result = await capability.run(
            {
                guild: { id: "g1" } as any,
                question: "what did alice say in reflexoes",
                currentChannelId: "c1",
            },
            {
                query: "what did alice say in reflexoes",
                authorId: "u-alice",
                channelIds: ["c-reflexoes"],
            }
        );

        expect(UnifiedMessageRetrieval.retrieve).toHaveBeenCalledWith(
            expect.objectContaining({
                question: "what did alice say in reflexoes",
                authorId: "u-alice",
                channelIds: ["c-reflexoes"],
            })
        );
        expect(result.tool).toBe("retrieve_messages");
        expect(result.summary).toContain("partial history evidence");
        expect(result.data).toMatchObject({
            targetAuthorId: "u-alice",
            targetChannelIds: ["c-reflexoes"],
            cacheHit: true,
        });
    });

    it("executes get_member_profile with an exact requester id", async () => {
        vi.spyOn(DiscordLiveService, "getMemberProfile").mockResolvedValue({
            id: "u-requester",
            username: "req",
            displayName: "Requester",
            globalName: null,
            nickname: null,
            roles: ["Admin"],
            joinedAt: null,
            joinedTimestamp: null,
            accountCreatedAt: null,
            avatarUrl: null,
            premiumSince: null,
            pending: false,
            bannerUrl: null,
            accentColor: null,
            bio: null,
        });

        const capability = CapabilityRegistry.get("get_member_profile");
        const result = await capability.run(
            {
                guild: { id: "g1" } as any,
                question: "Quem sou eu?",
            },
            {
                nameOrId: "u-requester",
            }
        );

        expect(DiscordLiveService.getMemberProfile).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" }),
            "u-requester"
        );
        expect(result.tool).toBe("get_member_profile");
        expect(result.summary).toContain("Requester");
        expect(result.data).toMatchObject({
            id: "u-requester",
            displayName: "Requester",
        });
    });

    it("executes resolve_member_identity with an exact id", async () => {
        vi.spyOn(DiscordLiveService, "resolveMemberIdentity").mockResolvedValue({
            query: "u-requester",
            resolvedId: "u-requester",
            displayName: "Requester",
            username: "req",
            globalName: null,
            nickname: null,
            isBot: false,
            isCurrentGuildMember: true,
            source: "live_id",
            confidence: "exact",
            roles: ["Admin"],
        });

        const capability = CapabilityRegistry.get("resolve_member_identity");
        const result = await capability.run(
            {
                guild: { id: "g1" } as any,
                question: "who is u-requester",
            },
            { queries: ["u-requester"] }
        );

        expect(DiscordLiveService.resolveMemberIdentity).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" }),
            "u-requester"
        );
        expect(result.tool).toBe("resolve_member_identity");
        expect(result.data).toMatchObject({
            results: [expect.objectContaining({ resolvedId: "u-requester", source: "live_id" })],
        });
    });

    it("executes resolve_channel_targets through the shared guild discovery service", async () => {
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargetsBatch").mockResolvedValue([
            {
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
            },
        ]);

        const capability = CapabilityRegistry.get("resolve_channel_targets");
        const result = await capability.run(
            {
                guild: { id: "g1" } as any,
                question: "look in reflexoes",
                currentChannelId: "c1",
            },
            { targets: ["reflexoes"] }
        );

        expect(DiscordGuildDiscoveryService.resolveChannelTargetsBatch).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" }),
            ["reflexoes"],
            "c1"
        );
        expect(result.tool).toBe("resolve_channel_targets");
        expect(result.data).toMatchObject({
            results: [expect.objectContaining({ resolvedIds: ["c-reflexoes"], confidence: "high" })],
        });
    });

    it("executes list_guild_structure through the shared guild discovery service", async () => {
        vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
            {
                id: "cat-1",
                guildId: "g1",
                name: "General",
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
                id: "c-general",
                guildId: "g1",
                name: "general",
                type: "0",
                parentCategoryId: "cat-1",
                parentCategoryName: "General",
                isReadable: true,
                isViewable: true,
                isIndexed: true,
                source: "live",
                missingOrDeletedPossible: false,
            },
        ]);

        const capability = CapabilityRegistry.get("list_guild_structure");
        const result = await capability.run(
            {
                guild: { id: "g1" } as any,
                question: "show me the guild structure",
            },
            {}
        );

        expect(DiscordGuildDiscoveryService.listGuildStructure).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" })
        );
        expect(result.tool).toBe("list_guild_structure");
        expect(result.data).toMatchObject({
            entries: [expect.objectContaining({ id: "cat-1" }), expect.objectContaining({ id: "c-general" })],
        });
    });

    it("executes list_members through the live member service", async () => {
        vi.spyOn(DiscordLiveService, "listMembers").mockResolvedValue({
            members: [
                {
                    id: "u1",
                    username: "alice",
                    displayName: "Alice",
                    joinedTimestamp: 100,
                },
            ],
            totalCount: 1,
            returnedCount: 1,
            hasMore: false,
            offset: 0,
            limit: 10,
            sort: "joined_at",
            filters: "alice",
        });

        const capability = CapabilityRegistry.get("list_members");
        const result = await capability.run(
            {
                guild: { id: "g1" } as any,
                question: "list alice",
            },
            {
                filters: "alice",
                limit: 10,
                offset: 0,
            }
        );

        expect(DiscordLiveService.listMembers).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" }),
            expect.objectContaining({
                filters: "alice",
                limit: 10,
                offset: 0,
                sort: "joined_at",
            })
        );
        expect(result.tool).toBe("list_members");
        expect(result.data).toMatchObject({
            returnedCount: 1,
            members: [expect.objectContaining({ id: "u1" })],
        });
    });

    it("executes get_guild_context through the live guild service", async () => {
        vi.spyOn(DiscordLiveService, "getGuildContext").mockResolvedValue({
            id: "g1",
            name: "Oz Synthesis",
            memberCount: 42,
            channelCount: 12,
        });

        const capability = CapabilityRegistry.get("get_guild_context");
        const result = await capability.run(
            {
                guild: { id: "g1" } as any,
                question: "how big is this guild?",
            },
            {}
        );

        expect(DiscordLiveService.getGuildContext).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" })
        );
        expect(result.tool).toBe("get_guild_context");
        expect(result.data).toMatchObject({
            name: "Oz Synthesis",
            memberCount: 42,
            channelCount: 12,
        });
    });
});

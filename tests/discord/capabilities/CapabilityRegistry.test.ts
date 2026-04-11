import { beforeEach, describe, expect, it, vi } from "vitest";
import { CapabilityRegistry } from "@/discord/capabilities/CapabilityRegistry";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";

describe("CapabilityRegistry", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("executes retrieve_messages through the unified retrieval pipeline", async () => {
        vi.spyOn(UnifiedMessageRetrieval, "retrieve").mockResolvedValue({
            query: "what did alice say in reflexoes",
            results: [
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
            sourceOrigin: "cache",
            targetAuthorId: "u-alice",
            targetChannelIds: ["c-reflexoes"],
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
                channelIds: "c-reflexoes",
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
        expect(result.summary).toContain("partial message evidence");
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
            { query: "u-requester" }
        );

        expect(DiscordLiveService.resolveMemberIdentity).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" }),
            "u-requester"
        );
        expect(result.tool).toBe("resolve_member_identity");
        expect(result.data).toMatchObject({
            resolvedId: "u-requester",
            source: "live_id",
        });
    });

    it("executes resolve_channel_targets through the shared guild discovery service", async () => {
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

        const capability = CapabilityRegistry.get("resolve_channel_targets");
        const result = await capability.run(
            {
                guild: { id: "g1" } as any,
                question: "look in reflexoes",
                currentChannelId: "c1",
            },
            { targetText: "reflexoes" }
        );

        expect(DiscordGuildDiscoveryService.resolveChannelTargets).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" }),
            "reflexoes",
            "c1"
        );
        expect(result.tool).toBe("resolve_channel_targets");
        expect(result.data).toMatchObject({
            resolvedIds: ["c-reflexoes"],
            confidence: "high",
        });
    });
});

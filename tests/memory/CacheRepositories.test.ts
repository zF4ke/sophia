import { beforeEach, describe, expect, it } from "vitest";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { MemoryDatabase } from "@/memory/MemoryDatabase";

describe("cache repositories", () => {
    beforeEach(() => {
        process.env.SOPHIA_MEMORY_DB_PATH = ":memory:";
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        MemoryDatabase.reset();
    });

    it("prefers same-channel reusable contexts over other guild channels", () => {
        const now = Date.now();
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c-other",
            channelScopeKey: "c-other",
            questionFingerprint: "roadmap release",
            routeIntent: "broad_search",
            evidenceText: "evidence other",
            citations: [],
            toolRuns: [],
            sufficient: true,
            groundingDecisionMode: "judge",
            createdTimestamp: now - 1000,
            expiryTimestamp: now + 60_000,
            createdResponseOrdinal: 1,
        });
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c-current",
            channelScopeKey: "c-current",
            questionFingerprint: "roadmap release",
            routeIntent: "broad_search",
            evidenceText: "evidence current",
            citations: [],
            toolRuns: [],
            sufficient: true,
            groundingDecisionMode: "judge",
            createdTimestamp: now,
            expiryTimestamp: now + 60_000,
            createdResponseOrdinal: 2,
        });

        const result = DiscordMemoryService.getReusableGroundedContext({
            guildId: "g1",
            currentChannelId: "c-current",
            questionFingerprint: "roadmap release",
            routeIntent: "broad_search",
            requireSufficient: true,
            currentResponseOrdinal: 3,
            maxResponsesAgo: 6,
        });

        expect(result?.channelId).toBe("c-current");
        expect(result?.evidenceText).toBe("evidence current");
    });

    it("falls back to guild-wide reusable context when there is no same-channel hit", () => {
        const now = Date.now();
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c-other",
            channelScopeKey: "c-other",
            questionFingerprint: "server rules",
            routeIntent: "broad_search",
            evidenceText: "guild evidence",
            citations: [],
            toolRuns: [],
            sufficient: true,
            groundingDecisionMode: "heuristic",
            createdTimestamp: now,
            expiryTimestamp: now + 60_000,
            createdResponseOrdinal: 2,
        });

        const result = DiscordMemoryService.getReusableGroundedContext({
            guildId: "g1",
            currentChannelId: "c-now",
            questionFingerprint: "server rules",
            routeIntent: "broad_search",
            requireSufficient: true,
            currentResponseOrdinal: 3,
            maxResponsesAgo: 6,
        });

        expect(result?.channelId).toBe("c-other");
        expect(result?.evidenceText).toBe("guild evidence");
    });

    it("prunes expired cache rows", () => {
        const now = Date.now();
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c1",
            channelScopeKey: "c1",
            questionFingerprint: "expired",
            routeIntent: "broad_search",
            evidenceText: "old",
            citations: [],
            toolRuns: [],
            sufficient: true,
            groundingDecisionMode: "judge",
            createdTimestamp: now - 10_000,
            expiryTimestamp: now - 1000,
            createdResponseOrdinal: 1,
        });

        DiscordMemoryService.pruneCacheEntries(now);

        const result = DiscordMemoryService.getReusableGroundedContext({
            guildId: "g1",
            currentChannelId: "c1",
            questionFingerprint: "expired",
            routeIntent: "broad_search",
            requireSufficient: true,
            currentResponseOrdinal: 2,
            maxResponsesAgo: 6,
        });

        expect(result).toBeNull();
    });

    it("ignores reusable contexts that are too many responses old", () => {
        const now = Date.now();
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c1",
            channelScopeKey: "c1",
            questionFingerprint: "stale by ordinal",
            routeIntent: "broad_search",
            evidenceText: "old but not expired by time",
            citations: [],
            toolRuns: [],
            sufficient: true,
            groundingDecisionMode: "judge",
            createdTimestamp: now,
            expiryTimestamp: now + 20 * 60_000,
            createdResponseOrdinal: 1,
        });

        const result = DiscordMemoryService.getReusableGroundedContext({
            guildId: "g1",
            currentChannelId: "c1",
            questionFingerprint: "stale by ordinal",
            routeIntent: "broad_search",
            requireSufficient: true,
            currentResponseOrdinal: 8,
            maxResponsesAgo: 6,
        });

        expect(result).toBeNull();
    });

    it("can recover the most recent reusable context for follow-up reuse without a question fingerprint", () => {
        const now = Date.now();
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c-other",
            channelScopeKey: "c-other",
            questionFingerprint: "older question",
            routeIntent: "person_target",
            evidenceText: "older evidence",
            citations: [],
            toolRuns: [],
            sufficient: true,
            groundingDecisionMode: "heuristic",
            createdTimestamp: now - 5_000,
            expiryTimestamp: now + 60_000,
            createdResponseOrdinal: 1,
        });
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c-now",
            channelScopeKey: "c-now",
            questionFingerprint: "newer question",
            routeIntent: "person_target",
            evidenceText: "newer evidence",
            citations: [],
            toolRuns: [],
            sufficient: true,
            groundingDecisionMode: "heuristic",
            createdTimestamp: now,
            expiryTimestamp: now + 60_000,
            createdResponseOrdinal: 2,
        });

        const result = DiscordMemoryService.getRecentReusableGroundedContext({
            guildId: "g1",
            currentChannelId: "c-now",
            routeIntent: "person_target",
            requireSufficient: true,
            currentResponseOrdinal: 3,
            maxResponsesAgo: 6,
        });

        expect(result?.channelId).toBe("c-now");
        expect(result?.evidenceText).toBe("newer evidence");
    });
});

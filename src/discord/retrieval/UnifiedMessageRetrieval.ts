import type { Guild } from "discord.js";
import {
    DiscordChannelCrawlService,
    INTERACTIVE_CRAWL_LIMIT,
} from "@/discord/live/DiscordChannelCrawlService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { RetrievedChunk } from "@/shared/appTypes";
import type { RetrievalSourceOrigin } from "@/runtime/contracts";

const MAX_CHANNEL_ESCALATIONS = 2;
const ESCALATION_FETCH_LIMIT = Math.min(150, INTERACTIVE_CRAWL_LIMIT);

function buildScopedPreviewResults(
    crawls: Awaited<ReturnType<typeof DiscordChannelCrawlService.crawlChannelMessages>>[],
    guildId: string | null,
    limit: number
): RetrievedChunk[] {
    return crawls
        .flatMap((crawl) =>
            (crawl.previewMessages || []).map<RetrievedChunk>((message) => ({
                messageId: message.messageId,
                channelId: crawl.channelId,
                channelName: crawl.channelName,
                guildId,
                authorId: message.authorId,
                authorName: message.authorName,
                content: message.content,
                createdTimestamp: message.createdTimestamp,
                jumpLink: message.jumpLink,
                lexicalScore: 2,
                semanticScore: 0,
                recencyScore: 0,
                totalScore: 2,
            }))
        )
        .slice(0, limit);
}

function getResultStrength(result: RetrievedChunk): "strong" | "weak" {
    if (result.lexicalScore >= 2) {
        return "strong";
    }
    if (result.lexicalScore >= 1 && result.content.trim().length >= 80) {
        return "strong";
    }
    return "weak";
}

function summarizeStrength(results: RetrievedChunk[]) {
    return results.reduce(
        (acc, result) => {
            if (getResultStrength(result) === "strong") {
                acc.strongResultCount += 1;
            } else {
                acc.weakResultCount += 1;
            }
            return acc;
        },
        { strongResultCount: 0, weakResultCount: 0 }
    );
}

function hasSufficientLocalHits(results: RetrievedChunk[]): boolean {
    const strength = summarizeStrength(results);
    return strength.strongResultCount >= 1;
}

export interface UnifiedRetrievalResult {
    query: string;
    results: RetrievedChunk[];
    cacheHit: boolean;
    liveEscalated: boolean;
    searchedChannelIds: string[];
    fetchedChannelIds: string[];
    cacheEnriched: boolean;
    evidenceSufficient: boolean;
    strongResultCount: number;
    weakResultCount: number;
    sourceOrigin: RetrievalSourceOrigin;
    targetAuthorId: string | null;
    targetChannelIds: string[];
}

export class UnifiedMessageRetrieval {
    public static async retrieve(options: {
        guild: Guild | null;
        question: string;
        currentChannelId?: string | null;
        channelIds?: string[];
        authorId?: string;
        limit?: number;
        onProgress?: (toolName: string, summary: string) => Promise<void> | void;
    }): Promise<UnifiedRetrievalResult> {
        const limit = Math.max(1, options.limit ?? 8);
        const scope = {
            guildId: options.guild?.id || null,
            channelIds: options.channelIds?.length ? options.channelIds : undefined,
            authorIds: options.authorId ? [options.authorId] : undefined,
        };

        let results = await DiscordMemoryService.searchMessagesAsync(options.question, scope, limit);
        const searchedChannelIds = [...(options.channelIds || [])];
        const fetchedChannelIds: string[] = [];
        let cacheEnriched = false;
        let sourceOrigin: RetrievalSourceOrigin = results.length ? "cache" : "none";
        const crawlResults: Awaited<ReturnType<typeof DiscordChannelCrawlService.crawlChannelMessages>>[] = [];

        if (!hasSufficientLocalHits(results) && options.guild) {
            const rankedChannels = options.channelIds?.length
                ? options.channelIds.map((channelId) => ({
                      channelId,
                      channelName: channelId,
                      guildId: options.guild?.id || null,
                      isIndexed: false,
                      matchSource: "live_name" as const,
                      lastIndexedTimestamp: null,
                  }))
                : await DiscordChannelCrawlService.rankCandidateChannels(
                      options.guild,
                      options.question,
                      options.currentChannelId
                  );

            const escalationTargets = options.channelIds?.length
                ? rankedChannels
                : rankedChannels.slice(0, MAX_CHANNEL_ESCALATIONS);

            for (const channel of escalationTargets) {
                if (!searchedChannelIds.includes(channel.channelId)) {
                    searchedChannelIds.push(channel.channelId);
                }

                const crawl = await DiscordChannelCrawlService.crawlChannelMessages(
                    options.guild,
                    channel.channelId,
                    ESCALATION_FETCH_LIMIT,
                    options.question,
                    options.onProgress
                );
                crawlResults.push(crawl);

                if (crawl.messagesFetched > 0) {
                    fetchedChannelIds.push(channel.channelId);
                    cacheEnriched = true;
                }
            }

            if (cacheEnriched) {
                await DiscordChannelCrawlService.waitForBackgroundIngest();
                results = await DiscordMemoryService.searchMessagesAsync(
                    options.question,
                    scope,
                    limit
                );
                if (!results.length && options.channelIds?.length) {
                    results = buildScopedPreviewResults(crawlResults, options.guild?.id || null, limit);
                }
                sourceOrigin = results.length ? "cache_after_refresh" : "live_refresh";
            }
        }

        const strength = summarizeStrength(results);

        return {
            query: options.question,
            results,
            cacheHit: results.length > 0,
            liveEscalated: fetchedChannelIds.length > 0,
            searchedChannelIds,
            fetchedChannelIds,
            cacheEnriched,
            evidenceSufficient: hasSufficientLocalHits(results),
            strongResultCount: strength.strongResultCount,
            weakResultCount: strength.weakResultCount,
            sourceOrigin,
            targetAuthorId: options.authorId || null,
            targetChannelIds: options.channelIds || [],
        };
    }
}

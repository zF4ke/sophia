import type { Guild } from "discord.js";
import {
    DiscordChannelCrawlService,
    INTERACTIVE_CRAWL_LIMIT,
} from "@/discord/live/DiscordChannelCrawlService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type {
    ChannelCrawlResult,
    MultiLaneRetrievalResult,
    RetrievedChunk,
    RetrievalMode,
} from "@/shared/appTypes";

const MAX_CHANNEL_ESCALATIONS = 2;
const ESCALATION_FETCH_LIMIT = Math.min(150, INTERACTIVE_CRAWL_LIMIT);

function dedupeChunks(rows: RetrievedChunk[]): RetrievedChunk[] {
    const seen = new Set<string>();
    return rows.filter((row) => {
        if (seen.has(row.messageId)) {
            return false;
        }
        seen.add(row.messageId);
        return true;
    });
}

function getResultStrength(result: RetrievedChunk): "strong" | "weak" {
    if (result.lexicalScore >= 2) {
        return "strong";
    }
    if (result.lexicalScore >= 1 && result.content.trim().length >= 80) {
        return "strong";
    }
    if (result.semanticScore >= 1 && result.content.trim().length >= 120) {
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

function buildHistoryPreviewResults(
    crawls: ChannelCrawlResult[],
    guildId: string | null,
    limit: number
): RetrievedChunk[] {
    return dedupeChunks(
        crawls.flatMap((crawl) =>
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
                lexicalScore: 0,
                semanticScore: 0,
                recencyScore: 1,
                totalScore: 1,
            }))
        )
    )
        .sort((left, right) => left.createdTimestamp - right.createdTimestamp)
        .slice(0, limit);
}

function buildCursorMap(
    channelIds: string[],
    historyMessages: RetrievedChunk[],
    previous: Record<string, string | null>
): Record<string, string | null> {
    const next = { ...previous };
    for (const channelId of channelIds) {
        const channelRows = historyMessages.filter((row) => row.channelId === channelId);
        if (channelRows.length) {
            next[channelId] = channelRows[0]?.messageId || null;
        } else if (!(channelId in next)) {
            next[channelId] = null;
        }
    }
    return next;
}

async function buildExhaustion(
    guildId: string | null,
    channelIds: string[],
    historyMessages: RetrievedChunk[],
    crawls: ChannelCrawlResult[]
): Promise<{ exhaustedChannelIds: string[]; exhausted: boolean }> {
    const exhaustedChannelIds: string[] = [];

    for (const channelId of channelIds) {
        const crawl = crawls.find((item) => item.channelId === channelId);
        if (crawl?.exhausted) {
            exhaustedChannelIds.push(channelId);
            continue;
        }

        const summary = await DiscordMemoryService.getChannelSummaryAsync(channelId);
        if (!summary) {
            continue;
        }

        const oldestReturned = historyMessages
            .filter((row) => row.channelId === channelId)
            .sort((left, right) => left.createdTimestamp - right.createdTimestamp)[0];

        if (
            oldestReturned &&
            summary.firstMessageTimestamp != null &&
            oldestReturned.createdTimestamp <= summary.firstMessageTimestamp
        ) {
            exhaustedChannelIds.push(channelId);
        }
    }

    return {
        exhaustedChannelIds,
        exhausted: channelIds.length > 0 && exhaustedChannelIds.length === channelIds.length,
    };
}

function buildCombinedResults(
    historyMessages: RetrievedChunk[],
    semanticMatches: RetrievedChunk[],
    limit: number
): RetrievedChunk[] {
    return dedupeChunks([...historyMessages, ...semanticMatches]).slice(0, Math.max(limit, historyMessages.length, semanticMatches.length));
}

function shouldEscalateLive(
    mode: RetrievalMode,
    historyMessages: RetrievedChunk[],
    semanticMatches: RetrievedChunk[],
    channelIds: string[]
): boolean {
    if (!channelIds.length) {
        return !semanticMatches.length;
    }

    if (mode === "history") {
        return historyMessages.length === 0;
    }
    if (mode === "semantic") {
        return summarizeStrength(semanticMatches).strongResultCount === 0;
    }
    return historyMessages.length === 0 && summarizeStrength(semanticMatches).strongResultCount === 0;
}

export class UnifiedMessageRetrieval {
    public static async retrieve(options: {
        guild: Guild | null;
        question: string;
        currentChannelId?: string | null;
        channelIds?: string[];
        authorId?: string;
        limit?: number;
        mode?: RetrievalMode;
        beforeTimestamp?: number | null;
        afterTimestamp?: number | null;
        cursor?: Record<string, string | null>;
        excludedMessageIds?: string[];
        onProgress?: (toolName: string, summary: string) => Promise<void> | void;
    }): Promise<MultiLaneRetrievalResult> {
        const limit = Math.max(1, options.limit ?? 8);
        const mode = options.mode ?? (options.channelIds?.length || options.currentChannelId ? "history" : "mixed");
        const scopedChannelIds =
            options.channelIds?.length
                ? options.channelIds
                : options.currentChannelId
                  ? [options.currentChannelId]
                  : [];
        const scope = {
            guildId: options.guild?.id || null,
            channelIds: scopedChannelIds.length ? scopedChannelIds : undefined,
            authorIds: options.authorId ? [options.authorId] : undefined,
            beforeTimestamp: options.beforeTimestamp ?? undefined,
            afterTimestamp: options.afterTimestamp ?? undefined,
            excludedMessageIds: options.excludedMessageIds,
        };

        let historyMessages =
            mode === "semantic"
                ? []
                : (await DiscordMemoryService.getChannelHistoryPageAsync({
                      guildId: options.guild?.id || null,
                      channelIds: scopedChannelIds,
                      authorId: options.authorId || null,
                      beforeTimestamp: options.beforeTimestamp ?? null,
                      afterTimestamp: options.afterTimestamp ?? null,
                      perChannelOldestMessageId: options.cursor,
                      excludedMessageIds: options.excludedMessageIds,
                      limit,
                  })).map<RetrievedChunk>((message) => ({
                      messageId: message.id,
                      channelId: message.channelId,
                      channelName: message.channelName,
                      guildId: message.guildId,
                      authorId: message.authorId,
                      authorName: message.authorName,
                      content: message.content,
                      createdTimestamp: message.createdTimestamp,
                      jumpLink: message.jumpLink,
                      lexicalScore: 0,
                      semanticScore: 0,
                      recencyScore: 1,
                      totalScore: 1,
                  }));
        let semanticMatches =
            mode === "history"
                ? []
                : await DiscordMemoryService.searchMessagesAsync(options.question, scope, limit);

        const searchedChannelIds = [...scopedChannelIds];
        const fetchedChannelIds: string[] = [];
        let cacheEnriched = false;
        let sourceOrigin: MultiLaneRetrievalResult["sourceOrigin"] =
            historyMessages.length || semanticMatches.length ? "cache" : "none";
        const crawlResults: ChannelCrawlResult[] = [];

        if (options.guild && shouldEscalateLive(mode, historyMessages, semanticMatches, scopedChannelIds)) {
            const rankedChannels = scopedChannelIds.length
                ? scopedChannelIds.map((channelId) => ({
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

            const escalationTargets = scopedChannelIds.length
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
                historyMessages =
                    mode === "semantic"
                        ? []
                        : (await DiscordMemoryService.getChannelHistoryPageAsync({
                              guildId: options.guild?.id || null,
                              channelIds: searchedChannelIds,
                              authorId: options.authorId || null,
                              beforeTimestamp: options.beforeTimestamp ?? null,
                              afterTimestamp: options.afterTimestamp ?? null,
                              perChannelOldestMessageId: options.cursor,
                              excludedMessageIds: options.excludedMessageIds,
                              limit,
                          })).map<RetrievedChunk>((message) => ({
                              messageId: message.id,
                              channelId: message.channelId,
                              channelName: message.channelName,
                              guildId: message.guildId,
                              authorId: message.authorId,
                              authorName: message.authorName,
                              content: message.content,
                              createdTimestamp: message.createdTimestamp,
                              jumpLink: message.jumpLink,
                              lexicalScore: 0,
                              semanticScore: 0,
                              recencyScore: 1,
                              totalScore: 1,
                          }));
                semanticMatches =
                    mode === "history"
                        ? []
                        : await DiscordMemoryService.searchMessagesAsync(
                              options.question,
                              {
                                  ...scope,
                                  channelIds: searchedChannelIds.length ? searchedChannelIds : undefined,
                              },
                              limit
                          );
                if (!historyMessages.length && searchedChannelIds.length) {
                    historyMessages = buildHistoryPreviewResults(
                        crawlResults,
                        options.guild?.id || null,
                        limit
                    );
                }
                sourceOrigin = historyMessages.length || semanticMatches.length ? "cache_after_refresh" : "live_refresh";
            }
        }

        const combinedResults = buildCombinedResults(historyMessages, semanticMatches, limit);
        const strength = summarizeStrength(mode === "history" ? historyMessages : combinedResults);
        const continuation = {
            perChannelOldestMessageId: buildCursorMap(
                searchedChannelIds,
                historyMessages,
                options.cursor || {}
            ),
            continuationAvailable: Boolean(historyMessages.length),
        };
        const exhaustion = await buildExhaustion(
            options.guild?.id || null,
            searchedChannelIds,
            historyMessages,
            crawlResults
        );

        const evidenceSufficient =
            mode === "history"
                ? historyMessages.length > 0
                : mode === "semantic"
                  ? summarizeStrength(semanticMatches).strongResultCount > 0
                  : historyMessages.length > 0 || summarizeStrength(semanticMatches).strongResultCount > 0;

        return {
            query: options.question,
            mode,
            historyMessages: dedupeChunks(historyMessages),
            semanticMatches: dedupeChunks(semanticMatches),
            combinedResults,
            cacheHit: Boolean(historyMessages.length || semanticMatches.length),
            liveEscalated: fetchedChannelIds.length > 0,
            searchedChannelIds,
            fetchedChannelIds,
            cacheEnriched,
            evidenceSufficient,
            strongResultCount: strength.strongResultCount,
            weakResultCount: strength.weakResultCount,
            historyMessageCount: historyMessages.length,
            semanticMatchCount: semanticMatches.length,
            sourceOrigin,
            targetAuthorId: options.authorId || null,
            targetChannelIds: searchedChannelIds,
            continuation,
            exhaustion,
            accumulatedWindow: {
                beforeTimestamp: options.beforeTimestamp ?? null,
                afterTimestamp: options.afterTimestamp ?? null,
            },
            beforeTimestamp: options.beforeTimestamp ?? null,
            afterTimestamp: options.afterTimestamp ?? null,
            excludedMessageIds: options.excludedMessageIds || [],
        };
    }
}

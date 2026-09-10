import type { Guild } from "discord.js";
import { getAppConfig } from "@/app/AppConfig";
import { SettingsService } from "@/app/SettingsService";
import {
    DiscordChannelCrawlService,
} from "@/discord/live/DiscordChannelCrawlService";
import { DiscordBackfillCrawler } from "@/discord/live/DiscordBackfillCrawler";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type {
    ChannelCrawlResult,
    MultiLaneRetrievalResult,
    RetrievedChunk,
    RetrievalMode,
    SemanticContinuationCursor,
} from "@/shared/appTypes";

const MAX_CHANNEL_ESCALATIONS = 2;

function getEscalationFetchLimit(): number {
    return Math.max(1, getAppConfig().runtime.escalationFetchLimit);
}

function edgePrefetchEnabled(): boolean {
    try {
        return SettingsService.load().runtime.edgePrefetch !== false;
    } catch {
        return true;
    }
}

/**
 * Queue deep backfill for channels whose history page touched the indexed
 * boundary (oldest retrieved snowflake <= crawl checkpoint). Fire-and-forget —
 * called with `void` after a history fetch; errors are swallowed.
 */
async function edgePrefetch(channelIds: string[], historyMessages: RetrievedChunk[]): Promise<void> {
    const unique = [...new Set(channelIds)];
    for (const channelId of unique) {
        const [crawlState] = await DiscordMemoryService.getChannelCrawlStateAsync(channelId);
        if (!crawlState || crawlState.exhausted) continue;
        const boundary = crawlState.oldestFetchedMessageId;
        if (!boundary) continue;
        const oldestInPage = historyMessages
            .filter((m) => m.channelId === channelId)
            .reduce<RetrievedChunk | null>((oldest, m) => {
                if (!oldest) return m;
                return BigInt(m.messageId) < BigInt(oldest.messageId) ? m : oldest;
            }, null);
        if (!oldestInPage) continue;
        // Page did not reach the boundary — plenty of indexed history left.
        if (BigInt(oldestInPage.messageId) > BigInt(boundary)) continue;
        await DiscordBackfillCrawler.enqueue(channelId, {
            reason: "edge_prefetch",
            priority: 5,
            guildId: oldestInPage.guildId,
        });
    }
}

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
                authorUsername: message.authorUsername ?? null,
                authorNickname: message.authorNickname ?? null,
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
    previous: Record<string, string | null>,
    crawlOldestByChannel: Record<string, string | null>,
    order: "newest" | "oldest" = "newest"
): Record<string, string | null> {
    const next = { ...previous };
    for (const channelId of channelIds) {
        const channelRows = historyMessages.filter((row) => row.channelId === channelId);
        if (channelRows.length) {
            // Newest-first: cursor = oldest row in page (first, since results are chronological).
            // Oldest-first: cursor = newest row in page (last, since results are chronological).
            const boundary = order === "oldest"
                ? channelRows[channelRows.length - 1]
                : channelRows[0];
            next[channelId] = boundary?.messageId || null;
        } else if (crawlOldestByChannel[channelId]) {
            next[channelId] = crawlOldestByChannel[channelId];
        } else if (!(channelId in next)) {
            next[channelId] = null;
        }
    }
    return next;
}

function buildSemanticCursor(results: RetrievedChunk[]): SemanticContinuationCursor | null {
    const last = results[results.length - 1];
    if (!last) {
        return null;
    }

    return {
        lastScore: last.totalScore,
        lastCreatedTimestamp: last.createdTimestamp,
        lastMessageId: last.messageId,
    };
}

async function buildExhaustion(
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

function buildLaneExhaustion(
    historyExhaustion: { exhaustedChannelIds: string[]; exhausted: boolean },
    semanticExhausted: boolean
) {
    return {
        historyExhaustedChannelIds: historyExhaustion.exhaustedChannelIds,
        historyExhausted: historyExhaustion.exhausted,
        semanticExhausted,
        exhaustedChannelIds: historyExhaustion.exhaustedChannelIds,
        exhausted:
            historyExhaustion.exhausted &&
            semanticExhausted,
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
        order?: "newest" | "oldest";
        beforeTimestamp?: number | null;
        afterTimestamp?: number | null;
        aroundMessageId?: string;
        cursor?: {
            history?: Record<string, string | null>;
            semantic?: SemanticContinuationCursor | null;
        };
        excludedMessageIds?: string[];
        onProgress?: (toolName: string, summary: string) => Promise<void> | void;
    }): Promise<MultiLaneRetrievalResult> {
        // --- Around-message shortcut: fetch context around a known message ---
        if (options.aroundMessageId) {
            const contextWindow = getAppConfig().runtime.retrievalContextWindow;
            let contextMessages = await DiscordMemoryService.getMessageThreadAsync(
                options.aroundMessageId,
                contextWindow
            );

            // If the message isn't cached, attempt a targeted live crawl
            if (!contextMessages.length && options.guild) {
                // We don't know the channel, so try scoped channels or skip
                const channelIds = options.channelIds?.length
                    ? options.channelIds
                    : options.currentChannelId
                      ? [options.currentChannelId]
                      : [];
                for (const channelId of channelIds) {
                    const crawl = await DiscordChannelCrawlService.crawlChannelMessages(
                        options.guild,
                        channelId,
                        getEscalationFetchLimit(),
                        options.question,
                        options.onProgress
                    );
                    if (crawl.messagesFetched > 0) {
                        await DiscordChannelCrawlService.waitForBackgroundIngest();
                        contextMessages = await DiscordMemoryService.getMessageThreadAsync(
                            options.aroundMessageId,
                            contextWindow
                        );
                        if (contextMessages.length) break;
                    }
                }
            }

            const guildId = options.guild?.id || null;
            const historyMessages: RetrievedChunk[] = contextMessages.map((message) => ({
                messageId: message.id,
                channelId: message.channelId,
                channelName: message.channelName,
                guildId,
                authorId: message.authorId,
                authorName: message.authorName,
                authorUsername: message.authorUsername ?? null,
                authorNickname: message.authorNickname ?? null,
                content: message.content,
                createdTimestamp: message.createdTimestamp,
                jumpLink: message.jumpLink,
                lexicalScore: 0,
                semanticScore: 0,
                recencyScore: 1,
                totalScore: 1,
            }));
            const strength = summarizeStrength(historyMessages);
            const channelIds = [...new Set(historyMessages.map((m) => m.channelId))];
            return {
                query: options.question,
                mode: "history",
                historyMessages,
                semanticMatches: [],
                combinedResults: historyMessages,
                cacheHit: historyMessages.length > 0,
                liveEscalated: false,
                searchedChannelIds: channelIds,
                fetchedChannelIds: [],
                cacheEnriched: false,
                evidenceSufficient: historyMessages.length > 0,
                strongResultCount: strength.strongResultCount,
                weakResultCount: strength.weakResultCount,
                historyMessageCount: historyMessages.length,
                semanticMatchCount: 0,
                sourceOrigin: historyMessages.length ? "cache" : "none",
                targetAuthorId: options.authorId || null,
                targetChannelIds: channelIds,
                continuation: {
                    history: { perChannelOldestMessageId: {}, continuationAvailable: false },
                    semantic: { cursor: null, continuationAvailable: false },
                    perChannelOldestMessageId: {},
                    continuationAvailable: false,
                },
                exhaustion: {
                    historyExhaustedChannelIds: [],
                    historyExhausted: true,
                    semanticExhausted: true,
                    exhaustedChannelIds: [],
                    exhausted: true,
                },
                accumulatedWindow: {
                    beforeTimestamp: null,
                    afterTimestamp: null,
                },
                accumulatedUniqueCount: historyMessages.length,
                beforeTimestamp: null,
                afterTimestamp: null,
                excludedMessageIds: [],
                retrievalDiagnostics: {
                    strictScopedQuery: false,
                    continuationInputsApplied: false,
                    scopedEmptyRetryAttempted: false,
                    scopedEmptyRetryRecovered: false,
                    retryStrategy: "none",
                },
            };
        }

        const limit = Math.max(1, options.limit ?? 20);
        const order: "newest" | "oldest" = options.order === "oldest" ? "oldest" : "newest";
        const mode = options.mode ?? (options.channelIds?.length || options.currentChannelId ? "history" : "mixed");
        const scopedChannelIds =
            options.channelIds?.length
                ? options.channelIds
                : options.currentChannelId
                  ? [options.currentChannelId]
                  : [];
        const historyCursor = options.cursor?.history || {};
        const semanticCursor = options.cursor?.semantic || null;
        const hasHistoryCursor = Object.keys(historyCursor).length > 0;
        const hasExcludedMessageIds = Boolean(options.excludedMessageIds?.length);
        const strictScopedQuery =
            options.authorId != null ||
            options.beforeTimestamp != null ||
            options.afterTimestamp != null;
        const scope = {
            guildId: options.guild?.id || null,
            channelIds: scopedChannelIds.length ? scopedChannelIds : undefined,
            authorIds: options.authorId ? [options.authorId] : undefined,
            beforeTimestamp: options.beforeTimestamp ?? undefined,
            afterTimestamp: options.afterTimestamp ?? undefined,
            excludedMessageIds: options.excludedMessageIds,
            semanticCursor,
        };

        async function fetchHistoryMessages(fetchOpts: {
            guildId: string | null;
            channelIds: string[];
            authorId: string | null;
            beforeTimestamp: number | null;
            afterTimestamp: number | null;
            perChannelOldestMessageId: Record<string, string | null>;
            excludedMessageIds?: string[];
            limit: number;
            order?: "newest" | "oldest";
        }): Promise<RetrievedChunk[]> {
            return (await DiscordMemoryService.getChannelHistoryPageAsync(fetchOpts)).map<RetrievedChunk>((message) => ({
                messageId: message.id,
                channelId: message.channelId,
                channelName: message.channelName,
                guildId: message.guildId,
                authorId: message.authorId,
                authorName: message.authorName,
                authorUsername: message.authorUsername ?? null,
                authorNickname: message.authorNickname ?? null,
                content: message.content,
                createdTimestamp: message.createdTimestamp,
                jumpLink: message.jumpLink,
                lexicalScore: 0,
                semanticScore: 0,
                recencyScore: 1,
                totalScore: 1,
            }));
        }

        let historyMessages =
            mode === "semantic"
                ? []
                : await fetchHistoryMessages({
                      guildId: options.guild?.id || null,
                      channelIds: scopedChannelIds,
                      authorId: options.authorId || null,
                      beforeTimestamp: options.beforeTimestamp ?? null,
                      afterTimestamp: options.afterTimestamp ?? null,
                      perChannelOldestMessageId: historyCursor,
                      excludedMessageIds: options.excludedMessageIds,
                      limit,
                      order,
                  });
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
        let retryStrategy: "none" | "without_excluded" | "without_cursor" = "none";
        let scopedEmptyRetryAttempted = false;
        let scopedEmptyRetryRecovered = false;

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
                    getEscalationFetchLimit(),
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
                        : await fetchHistoryMessages({
                              guildId: options.guild?.id || null,
                              channelIds: searchedChannelIds,
                              authorId: options.authorId || null,
                              beforeTimestamp: options.beforeTimestamp ?? null,
                              afterTimestamp: options.afterTimestamp ?? null,
                              perChannelOldestMessageId: historyCursor,
                              excludedMessageIds: options.excludedMessageIds,
                              limit,
                              order,
                          });
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
                const allowPreviewFallback =
                    options.authorId == null &&
                    options.beforeTimestamp == null &&
                    options.afterTimestamp == null;
                if (!historyMessages.length && searchedChannelIds.length && allowPreviewFallback) {
                    historyMessages = buildHistoryPreviewResults(
                        crawlResults,
                        options.guild?.id || null,
                        limit
                    );
                }
                sourceOrigin = historyMessages.length || semanticMatches.length ? "cache_after_refresh" : "live_refresh";
            }

            // Targeted time-scoped crawl: if we have a time range but the standard
            // crawl (which fetches from the latest messages) didn't reach the target
            // period, do a second crawl starting from the target timestamp.
            const TARGETED_CRAWL_LIMIT = 100;
            if (
                !historyMessages.length &&
                mode !== "semantic" &&
                options.beforeTimestamp != null &&
                searchedChannelIds.length &&
                options.guild?.channels
            ) {
                for (const channelId of searchedChannelIds) {
                    const targetedCrawl = await DiscordChannelCrawlService.crawlChannelMessagesAtTime(
                        options.guild,
                        channelId,
                        options.beforeTimestamp,
                        TARGETED_CRAWL_LIMIT,
                        options.question,
                        options.onProgress
                    );
                    crawlResults.push(targetedCrawl);
                    if (targetedCrawl.messagesFetched > 0) {
                        if (!fetchedChannelIds.includes(channelId)) {
                            fetchedChannelIds.push(channelId);
                        }
                        cacheEnriched = true;
                    }
                }

                if (cacheEnriched) {
                    await DiscordChannelCrawlService.waitForBackgroundIngest();
                    historyMessages = await fetchHistoryMessages({
                        guildId: options.guild?.id || null,
                        channelIds: searchedChannelIds,
                        authorId: options.authorId || null,
                        beforeTimestamp: options.beforeTimestamp ?? null,
                        afterTimestamp: options.afterTimestamp ?? null,
                        perChannelOldestMessageId: historyCursor,
                        excludedMessageIds: options.excludedMessageIds,
                        limit,
                        order,
                    });
                    if (!semanticMatches.length && mode !== "history") {
                        semanticMatches = await DiscordMemoryService.searchMessagesAsync(
                            options.question,
                            {
                                ...scope,
                                channelIds: searchedChannelIds.length ? searchedChannelIds : undefined,
                            },
                            limit
                        );
                    }
                    sourceOrigin = historyMessages.length || semanticMatches.length ? "cache_after_refresh" : "live_refresh";
                }
            }

            // Author-targeted deep crawl: when filtering by authorId but the
            // standard crawl didn't reach any of their messages, continue
            // crawling deeper.  crawlChannelMessages picks up from the saved
            // cursor so each call extends further into history.
            const AUTHOR_DEEP_CRAWL_MAX_PASSES = 3;
            if (
                !historyMessages.length &&
                options.authorId &&
                mode !== "semantic" &&
                searchedChannelIds.length &&
                options.guild?.channels
            ) {
                for (let pass = 0; pass < AUTHOR_DEEP_CRAWL_MAX_PASSES && !historyMessages.length; pass++) {
                    let passEnriched = false;
                    for (const channelId of searchedChannelIds) {
                        const priorCrawl = crawlResults.filter((c) => c.channelId === channelId);
                        if (priorCrawl.some((c) => c.exhausted)) continue;

                        const deepCrawl = await DiscordChannelCrawlService.crawlChannelMessages(
                            options.guild,
                            channelId,
                            getEscalationFetchLimit(),
                            options.question,
                            options.onProgress
                        );
                        crawlResults.push(deepCrawl);
                        if (deepCrawl.messagesFetched > 0) {
                            if (!fetchedChannelIds.includes(channelId)) {
                                fetchedChannelIds.push(channelId);
                            }
                            passEnriched = true;
                        }
                    }

                    if (!passEnriched) break;

                    await DiscordChannelCrawlService.waitForBackgroundIngest();
                    historyMessages = await fetchHistoryMessages({
                        guildId: options.guild?.id || null,
                        channelIds: searchedChannelIds,
                        authorId: options.authorId || null,
                        beforeTimestamp: options.beforeTimestamp ?? null,
                        afterTimestamp: options.afterTimestamp ?? null,
                        perChannelOldestMessageId: historyCursor,
                        excludedMessageIds: options.excludedMessageIds,
                        limit,
                        order,
                    });
                    sourceOrigin = historyMessages.length ? "cache_after_refresh" : "live_refresh";
                }
            }
        }

        if (mode !== "semantic" && strictScopedQuery && !historyMessages.length) {
            if (hasExcludedMessageIds || hasHistoryCursor) {
                scopedEmptyRetryAttempted = true;
            }

            if (hasExcludedMessageIds) {
                const retriedWithoutExcluded = await fetchHistoryMessages({
                    guildId: options.guild?.id || null,
                    channelIds: searchedChannelIds,
                    authorId: options.authorId || null,
                    beforeTimestamp: options.beforeTimestamp ?? null,
                    afterTimestamp: options.afterTimestamp ?? null,
                    perChannelOldestMessageId: historyCursor,
                    limit,
                    order,
                });
                if (retriedWithoutExcluded.length) {
                    historyMessages = retriedWithoutExcluded;
                    retryStrategy = "without_excluded";
                    scopedEmptyRetryRecovered = true;
                }
            }

            if (!historyMessages.length && hasHistoryCursor) {
                const retriedWithoutCursor = await fetchHistoryMessages({
                    guildId: options.guild?.id || null,
                    channelIds: searchedChannelIds,
                    authorId: options.authorId || null,
                    beforeTimestamp: options.beforeTimestamp ?? null,
                    afterTimestamp: options.afterTimestamp ?? null,
                    perChannelOldestMessageId: {},
                    limit,
                    order,
                });
                if (retriedWithoutCursor.length) {
                    historyMessages = retriedWithoutCursor;
                    retryStrategy = "without_cursor";
                    scopedEmptyRetryRecovered = true;
                }
            }
        }

        const combinedResults = buildCombinedResults(historyMessages, semanticMatches, limit);

        // Edge prefetch: when a history page reaches the indexed boundary
        // (oldest retrieved snowflake <= crawl checkpoint), queue that channel
        // for deep backfill so the next pagination finds more history instead
        // of a dead end. Fire-and-forget; the queue dedupes.
        if (edgePrefetchEnabled() && historyMessages.length) {
            void edgePrefetch(searchedChannelIds, historyMessages).catch(() => {});
        }

        const crawlOldestByChannel = Object.fromEntries(
            crawlResults.map((crawl) => [crawl.channelId, crawl.oldestFetchedMessageId || null])
        );
        const strength = summarizeStrength(mode === "history" ? historyMessages : combinedResults);
        const historyContinuationCursor = buildCursorMap(
            searchedChannelIds,
            historyMessages,
            historyCursor,
            crawlOldestByChannel,
            order
        );
        const semanticContinuationCursor = buildSemanticCursor(semanticMatches);
        const historyExhaustion = await buildExhaustion(
            searchedChannelIds,
            historyMessages,
            crawlResults
        );
        const semanticExhausted =
            mode === "history" ? true : semanticMatches.length < limit;
        const exhaustion = buildLaneExhaustion(historyExhaustion, semanticExhausted);
        const historyContinuationAvailable =
            !historyExhaustion.exhausted && (historyMessages.length > 0 || strictScopedQuery);
        const semanticContinuationAvailable =
            mode !== "history" && !semanticExhausted && semanticMatches.length > 0;
        const continuation = {
            history: {
                perChannelOldestMessageId: historyContinuationCursor,
                continuationAvailable: historyContinuationAvailable,
            },
            semantic: {
                cursor: semanticContinuationCursor,
                continuationAvailable: semanticContinuationAvailable,
            },
            perChannelOldestMessageId: historyContinuationCursor,
            continuationAvailable:
                historyContinuationAvailable || semanticContinuationAvailable,
        };

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
            accumulatedUniqueCount: new Set([
                ...(options.excludedMessageIds || []),
                ...combinedResults.map((row) => row.messageId),
            ]).size,
            beforeTimestamp: options.beforeTimestamp ?? null,
            afterTimestamp: options.afterTimestamp ?? null,
            excludedMessageIds: options.excludedMessageIds || [],
            retrievalDiagnostics: {
                strictScopedQuery,
                continuationInputsApplied: hasHistoryCursor || hasExcludedMessageIds,
                scopedEmptyRetryAttempted,
                scopedEmptyRetryRecovered,
                retryStrategy,
            },
        };
    }
}

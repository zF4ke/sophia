import type { Guild } from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import {
    DiscordChannelCrawlService,
    INTERACTIVE_CRAWL_LIMIT,
} from "@/discord/live/DiscordChannelCrawlService";
import type { ChannelCandidate, DiscordToolResult } from "@/shared/appTypes";

export async function listRelevantChannels(
    query: string,
    guild: Guild | null,
    currentChannelId?: string | null
): Promise<DiscordToolResult> {
    const memoryChannels = DiscordMemoryService.listRelevantChannels(query, {
        guildId: guild?.id || null,
    });
    const memoryChannelMap = new Map(
        memoryChannels.map((channel) => [channel.channelId, channel])
    );
    const rankedChannels = DiscordChannelCrawlService.rankCandidateChannels(
        guild,
        query,
        currentChannelId
    )
        .map((channel) => {
            const memoryMatch = memoryChannelMap.get(channel.channelId);
            return {
                channelId: channel.channelId,
                channelName: channel.channelName,
                hitCount: memoryMatch?.hitCount ?? 0,
                isIndexed: channel.isIndexed,
                matchSource: memoryMatch ? "memory" : channel.matchSource,
                lastIndexedTimestamp: channel.lastIndexedTimestamp,
            } satisfies ChannelCandidate;
        })
        .slice(0, 8);

    return {
        tool: "list_relevant_channels",
        summary: rankedChannels.length
            ? `Found ${rankedChannels.length} potentially relevant channels.`
            : "No relevant channels found in local memory.",
        data: rankedChannels,
    };
}

export async function crawlChannelMessages(
    guild: Guild | null,
    channelId: string,
    limit = INTERACTIVE_CRAWL_LIMIT,
    queryHint?: string,
    onProgress?: (toolName: string, summary: string) => Promise<void> | void
): Promise<DiscordToolResult> {
    const crawl = await DiscordChannelCrawlService.crawlChannelMessages(
        guild,
        channelId,
        limit,
        queryHint,
        onProgress
    );

    return {
        tool: "crawl_channel_messages",
        summary: crawl.messagesFetched
            ? crawl.backgroundIngestQueued
                ? `Fetched ${crawl.messagesFetched} messages from ${crawl.channelName}; queued ${crawl.messagesStored} for background indexing.`
                : `Fetched ${crawl.messagesFetched} messages from ${crawl.channelName} and stored ${crawl.messagesStored}.`
            : `No messages fetched from ${crawl.channelName}.`,
        data: crawl,
    };
}

import type {
    CategoryChannel,
    ChatInputCommandInteraction,
    TextChannel,
    ThreadChannel,
} from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { UIService } from "@/discord/ui/UIService";
import { EMOJIS } from "@/discord/constants";
import { DiscordHistoryReader } from "./DiscordHistoryReader";

export type IndexableChannel = TextChannel | ThreadChannel;

export interface FetchAndIngestBatchResult {
    ingested: number;
    nextBeforeId: string | null;
    reachedEnd: boolean;
    oldestTimestampInBatch: number | null;
}

/**
 * Fetch up to `batchSize` messages older than `beforeId` from Discord and
 * ingest them into local memory. Commits a checkpoint to `channel_crawl_state`
 * after the batch is persisted so a crash mid-crawl can resume from the last
 * committed `oldest_fetched_message_id`.
 *
 * Contract:
 *   - If `beforeId` is null/undefined, the caller should seed it from the
 *     channel's existing `oldest_fetched_message_id` (see `resumeBeforeId`).
 *   - `reachedEnd=true` means Discord returned 0 messages — the channel's
 *     crawl state is marked `exhausted=true`.
 */
export async function fetchAndIngestBatch(
    channel: IndexableChannel,
    beforeId: string | null | undefined,
    batchSize = 100,
    mode: "backfill" | "refresh" = "backfill"
): Promise<FetchAndIngestBatchResult> {
    const page = await DiscordHistoryReader.page(channel, { before: beforeId, limit: batchSize, mode });
    const batch = page.messages;

    if (!batch.size) {
        // Exhausted — nothing older exists.
        return {
            ingested: 0,
            nextBeforeId: page.before,
            reachedEnd: true,
            oldestTimestampInBatch: null,
        };
    }

    const messages = Array.from(batch.values()).sort(
        (a, b) => a.createdTimestamp - b.createdTimestamp
    );

    const oldestInBatch = messages[0];
    const nextBeforeId = oldestInBatch?.id ?? null;

    return {
        ingested: messages.length,
        nextBeforeId,
        reachedEnd: page.exhausted,
        oldestTimestampInBatch: oldestInBatch?.createdTimestamp ?? null,
    };
}

/**
 * Read the persisted resume point for a channel. Returns `null` if no crawl
 * state exists yet (first-time backfill).
 */
export async function resumeBeforeId(channelId: string): Promise<string | null> {
    const states = await DiscordMemoryService.getChannelCrawlStateAsync(channelId);
    const state = states[0];
    if (!state) return null;
    return state.oldestFetchedMessageId ?? null;
}

export class DiscordBackfillService {
    /**
     * Synchronously backfill up to `limit` messages. Resumes from the channel's
     * last `oldest_fetched_message_id`. Checkpoints after every batch, so
     * rerunning the same command continues where the previous run stopped
     * (or where a crash interrupted it).
     */
    public static async backfillChannel(
        channel: IndexableChannel,
        interaction?: ChatInputCommandInteraction,
        limit = 1000
    ): Promise<number> {
        let fetched = 0;
        let before: string | null = await resumeBeforeId(channel.id);

        while (fetched < limit) {
            const remaining = limit - fetched;
            const batchSize = Math.min(100, remaining);

            const result = await fetchAndIngestBatch(channel, before, batchSize);

            fetched += result.ingested;
            before = result.nextBeforeId;

            if (result.reachedEnd && result.ingested === 0) {
                break;
            }

            if (interaction) {
                await interaction.editReply(
                    UIService.formatStatusMessage(
                        EMOJIS.sync,
                        `Indexing ${channel.name}... ${fetched}/${limit}`
                    )
                );
            }

            if (result.ingested === 0) {
                break;
            }
        }

        return fetched;
    }

    public static async backfillCategory(
        category: CategoryChannel,
        interaction?: ChatInputCommandInteraction,
        limitPerChannel = 500
    ): Promise<{ channels: number; messages: number }> {
        let channels = 0;
        let messages = 0;

        const children = category.children.cache.filter((channel) =>
            this.isIndexableChannel(channel)
        );
        for (const child of children.values()) {
            channels += 1;
            messages += await this.backfillChannel(child, interaction, limitPerChannel);
        }

        return { channels, messages };
    }

    private static isIndexableChannel(channel: unknown): channel is IndexableChannel {
        return Boolean(channel) && typeof channel === "object" && "messages" in (channel as object);
    }
}

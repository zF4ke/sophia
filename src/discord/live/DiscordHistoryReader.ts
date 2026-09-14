import { Collection, type Message, type Guild } from "discord.js";
import { enrichMessagesWithGuildMembers } from "./enrichMessageAuthors";
import { KeyedLock } from "@/shared/KeyedLock";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

type Channel = { id: string; guild?: Guild; messages: { fetch(options: { limit: number; before?: string }): Promise<Collection<string, Message>> } };
/** Owns history-page fetch, ingestion and contiguous-history checkpoints across all crawlers. */
export class DiscordHistoryReader {
    private static readonly channels = new KeyedLock();
    static async page(channel: Channel, options: { limit: number; before?: string | null; mode: "backfill" | "refresh" | "target"; guild?: Guild | null }) {
        return this.channels.run(channel.id, async () => {
            const state = (await DiscordMemoryService.getChannelCrawlStateAsync(channel.id))[0];
            const before = options.mode === "backfill" ? state?.oldestFetchedMessageId ?? options.before : options.before;
            if (options.mode === "backfill" && state?.exhausted) return { messages: new Collection<string, Message>(), before: before ?? null, exhausted: true };
            const messages = await channel.messages.fetch({ limit: Math.max(1, Math.min(100, options.limit)), ...(before ? { before } : {}) });
            const ordered = [...messages.values()].sort((a, b) => a.createdTimestamp - b.createdTimestamp || a.id.localeCompare(b.id));
            await enrichMessagesWithGuildMembers(options.guild ?? channel.guild ?? null, ordered);
            for (const message of ordered) await DiscordMemoryService.ingestMessage(message);
            const nextBefore = ordered[0]?.id ?? before ?? null;
            // Only contiguous pages may move the global boundary. A targeted historical
            // sample cannot prove that the intervening history has been indexed.
            const contiguous = options.mode === "backfill" || options.mode === "refresh" &&
                (!state?.oldestFetchedMessageId || before === state.oldestFetchedMessageId);
            if (contiguous && !state?.exhausted) await DiscordMemoryService.updateChannelCrawlState(channel.id, nextBefore, messages.size === 0);
            return { messages, before: nextBefore, exhausted: messages.size === 0 };
        });
    }
}

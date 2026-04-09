import { TextChannel, ThreadChannel, type Guild } from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { ChannelCrawlResult } from "@/shared/appTypes";

type CrawlableChannel = TextChannel | ThreadChannel;
export const INTERACTIVE_CRAWL_LIMIT = 250;
const PREVIEW_MESSAGE_LIMIT = 12;

let backgroundIngestQueue: Promise<void> = Promise.resolve();

function normalizeLookupValue(value: string): string {
    return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, " ")
        .trim();
}

export class DiscordChannelCrawlService {
    private static enqueueBackgroundIngest(messages: Array<any>): void {
        if (!messages.length) {
            return;
        }

        backgroundIngestQueue = backgroundIngestQueue
            .then(async () => {
                for (const message of messages) {
                    await DiscordMemoryService.ingestMessage(message);
                }
            })
            .catch((error) => {
                console.error("Background crawl ingest failed:", error);
            });
    }

    public static listReadableChannels(guild: Guild | null): CrawlableChannel[] {
        if (!guild || !guild.channels?.cache) {
            return [];
        }

        return [...guild.channels.cache.values()]
            .filter((channel) =>
                (channel instanceof TextChannel || channel instanceof ThreadChannel) &&
                channel.viewable
            )
            .map((channel) => channel as CrawlableChannel)
            .sort((left, right) => left.name.localeCompare(right.name));
    }

    public static rankCandidateChannels(
        guild: Guild | null,
        query: string,
        currentChannelId?: string | null
    ): Array<{
        channelId: string;
        channelName: string;
        guildId: string | null;
        isIndexed: boolean;
        matchSource: "memory" | "live_name";
        lastIndexedTimestamp: number | null;
    }> {
        const normalizedTerms = query
            .toLowerCase()
            .split(/\s+/)
            .map((term) => term.trim())
            .filter((term) => term.length > 1);

        const knownChannels = new Map(
            DiscordMemoryService.getKnownChannels(guild?.id || null).map((channel) => [
                channel.channelId,
                channel,
            ])
        );

        const channels = this.listReadableChannels(guild).map((channel) => {
            const known = knownChannels.get(channel.id);
            const nameMatchScore = normalizedTerms.reduce((score, term) => {
                return score + (channel.name.toLowerCase().includes(term) ? 1 : 0);
            }, 0);

            return {
                channelId: channel.id,
                channelName: channel.name,
                guildId: guild?.id || null,
                isIndexed: Boolean(known),
                matchSource: (known ? "memory" : "live_name") as "memory" | "live_name",
                lastIndexedTimestamp: known?.lastSeenTimestamp || null,
                score:
                    (currentChannelId && channel.id === currentChannelId ? 3 : 0) +
                    nameMatchScore +
                    (known ? 1 : 0),
            };
        });

        return channels
            .sort((left, right) => {
                if (right.score !== left.score) {
                    return right.score - left.score;
                }

                return left.channelName.localeCompare(right.channelName);
            })
            .map(({ score, ...channel }) => channel);
    }

    public static resolveChannelIdsByName(
        guild: Guild | null,
        targetText: string,
        currentChannelId?: string | null
    ): string[] {
        const normalizedTarget = normalizeLookupValue(targetText);
        if (!normalizedTarget) {
            return [];
        }

        const candidates = this.listReadableChannels(guild)
            .map((channel) => {
                const normalizedName = normalizeLookupValue(channel.name);
                let score = 0;
                let matchedByName = false;

                if (normalizedName === normalizedTarget) {
                    score += 5;
                    matchedByName = true;
                }
                if (normalizedName.includes(normalizedTarget)) {
                    score += 3;
                    matchedByName = true;
                }
                if (normalizedTarget.includes(normalizedName)) {
                    score += 2;
                    matchedByName = true;
                }
                if (matchedByName && currentChannelId && channel.id === currentChannelId) {
                    score += 1;
                }

                return {
                    channelId: channel.id,
                    score,
                };
            })
            .filter((candidate) => candidate.score > 0)
            .sort((left, right) => right.score - left.score);

        return candidates.slice(0, 3).map((candidate) => candidate.channelId);
    }

    public static async crawlChannelMessages(
        guild: Guild | null,
        channelId: string,
        limit = INTERACTIVE_CRAWL_LIMIT,
        queryHint?: string,
        onProgress?: (toolName: string, summary: string) => Promise<void> | void
    ): Promise<ChannelCrawlResult> {
        if (!guild) {
            return {
                channelId,
                channelName: channelId,
                messagesFetched: 0,
                messagesStored: 0,
                exhausted: true,
                queryHint: queryHint || null,
                backgroundIngestQueued: false,
                previewMessages: [],
            };
        }

        const channel = guild.channels.cache.get(channelId);
        if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) {
            return {
                channelId,
                channelName: channel?.name || channelId,
                messagesFetched: 0,
                messagesStored: 0,
                exhausted: true,
                queryHint: queryHint || null,
                backgroundIngestQueued: false,
                previewMessages: [],
            };
        }

        DiscordMemoryService.upsertDiscoveredChannel(
            channel.id,
            guild.id,
            channel.name,
            Date.now()
        );

        const crawlState = DiscordMemoryService.getChannelCrawlState(channel.id)[0] || null;
        if (crawlState?.exhausted) {
            await onProgress?.(
                "crawl_channel_messages",
                `histórico já esgotado em ${channel.name}; evitando recrawl`
            );
            return {
                channelId: channel.id,
                channelName: channel.name,
                messagesFetched: 0,
                messagesStored: 0,
                exhausted: true,
                queryHint: queryHint || null,
                backgroundIngestQueued: false,
                previewMessages: [],
            };
        }

        let fetched = 0;
        let stored = 0;
        let before: string | undefined = crawlState?.oldestFetchedMessageId || undefined;
        let exhausted = false;
        const fetchedMessages: Array<any> = [];

        while (fetched < limit) {
            const batch = await channel.messages.fetch({
                limit: Math.min(100, limit - fetched),
                before,
            });

            if (!batch.size) {
                exhausted = true;
                break;
            }

            const messages = [...batch.values()].sort(
                (left, right) => left.createdTimestamp - right.createdTimestamp
            );
            fetchedMessages.push(...messages);

            fetched += batch.size;
            before = messages[0]?.id;
            stored = fetchedMessages.length;
            await onProgress?.(
                "crawl_channel_messages",
                `coletadas ${fetched}/${limit} mensagens de ${channel.name}`
            );
        }

        DiscordMemoryService.updateChannelCrawlState(channel.id, before || null, exhausted);
        const previewMessages = buildPreviewMessages(fetchedMessages, queryHint);
        this.enqueueBackgroundIngest(fetchedMessages);

        return {
            channelId: channel.id,
            channelName: channel.name,
            messagesFetched: fetched,
            messagesStored: stored,
            exhausted,
            queryHint: queryHint || null,
            backgroundIngestQueued: stored > 0,
            previewMessages,
        };
    }

    public static async waitForBackgroundIngest(): Promise<void> {
        await backgroundIngestQueue;
    }
}

function buildPreviewMessages(
    messages: Array<any>,
    queryHint?: string
): ChannelCrawlResult["previewMessages"] {
    const normalizedTerms = (queryHint || "")
        .split(/\s+/)
        .map((term) => normalizeLookupValue(term))
        .filter((term) => term.length > 1);

    const ranked = messages
        .filter((message) => Boolean(message?.content?.trim()))
        .map((message) => {
            const normalizedContent = normalizeLookupValue(String(message.content || ""));
            const score = normalizedTerms.reduce((total, term) => {
                return total + (normalizedContent.includes(term) ? 1 : 0);
            }, 0);

            return {
                message,
                score,
            };
        })
        .filter((item) => item.score > 0 || normalizedTerms.length === 0)
        .sort((left, right) => {
            if (right.score !== left.score) {
                return right.score - left.score;
            }

            return (right.message.createdTimestamp || 0) - (left.message.createdTimestamp || 0);
        })
        .slice(0, PREVIEW_MESSAGE_LIMIT);

    return ranked.map(({ message }) => ({
        messageId: String(message.id),
        authorId: String(message.author?.id || ""),
        authorName: String(message.author?.username || message.author?.displayName || "unknown"),
        content: String(message.content || ""),
        createdTimestamp: Number(message.createdTimestamp || 0),
        jumpLink:
            typeof message.url === "string"
                ? message.url
                : `https://discord.com/channels/${message.guildId}/${message.channelId}/${message.id}`,
    }));
}

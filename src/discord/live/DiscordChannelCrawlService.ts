import { ChannelType, Collection, type Guild, type Message } from "discord.js";
import { getAppConfig } from "@/app/AppConfig";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { ChannelCrawlResult } from "@/shared/appTypes";

type CrawlableChannel = {
    id: string;
    name: string;
    type: ChannelType | number | string;
    viewable?: boolean;
    parent?: {
        id?: string | null;
        name?: string | null;
        parent?: {
            id?: string | null;
            name?: string | null;
        } | null;
    } | null;
    messages: {
        fetch: (options: { limit: number; before?: string }) => Promise<any>;
    };
};

export function getInteractiveCrawlLimit(): number {
    return getAppConfig().runtime.interactiveCrawlLimit;
}

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

function resolveMessageAuthorIdentity(message: any): {
    authorName: string;
    authorUsername: string | null;
    authorNickname: string | null;
} {
    const guildMember = message.member || message.guild?.members?.cache?.get?.(message.author?.id) || null;
    const authorUsername =
        typeof message.author?.username === "string" && message.author.username.trim()
            ? message.author.username
            : null;
    const authorName =
        guildMember?.displayName ||
        (typeof message.author?.globalName === "string" && message.author.globalName.trim()
            ? message.author.globalName
            : null) ||
        authorUsername ||
        "unknown";

    return {
        authorName,
        authorUsername,
        authorNickname: guildMember?.nickname || null,
    };
}

async function enrichMessagesWithGuildMembers(guild: Guild | null, messages: Array<any>): Promise<void> {
    if (!guild?.members?.fetch) {
        return;
    }

    const memberPromises = new Map<string, Promise<any>>();

    for (const message of messages) {
        const authorId = typeof message?.author?.id === "string" ? message.author.id : null;
        if (!authorId || message.member) {
            continue;
        }

        if (!memberPromises.has(authorId)) {
            memberPromises.set(
                authorId,
                Promise.resolve(guild.members.cache?.get?.(authorId) ?? null).then(async (cached) => {
                    if (cached) {
                        return cached;
                    }

                    try {
                        return await guild.members.fetch(authorId);
                    } catch {
                        return null;
                    }
                })
            );
        }
    }

    if (!memberPromises.size) {
        return;
    }

    for (const message of messages) {
        const authorId = typeof message?.author?.id === "string" ? message.author.id : null;
        if (!authorId || message.member || !memberPromises.has(authorId)) {
            continue;
        }

        const resolved = await memberPromises.get(authorId);
        try {
            message.member = resolved;
        } catch {
            // Discord.js Message objects expose `member` as a getter-only
            // property on cached instances.  Fall back to a non-enumerable
            // shadow property so downstream code can still read it.
            Object.defineProperty(message, "member", {
                value: resolved,
                writable: true,
                configurable: true,
            });
        }
    }
}

function isThreadLike(channel: { type?: ChannelType | number | string } | null | undefined): boolean {
    return (
        channel?.type === ChannelType.PublicThread ||
        channel?.type === ChannelType.PrivateThread ||
        channel?.type === ChannelType.AnnouncementThread ||
        String(channel?.type) === String(ChannelType.PublicThread) ||
        String(channel?.type) === String(ChannelType.PrivateThread) ||
        String(channel?.type) === String(ChannelType.AnnouncementThread)
    );
}

function isCrawlableChannel(channel: unknown): channel is CrawlableChannel {
    if (!channel || typeof channel !== "object") {
        return false;
    }

    const candidate = channel as Partial<CrawlableChannel>;
    if (candidate.viewable === false) {
        return false;
    }

    const type = candidate.type;
    const isSupportedType =
        type === ChannelType.GuildText ||
        String(type) === String(ChannelType.GuildText) ||
        type === ChannelType.GuildVoice ||
        String(type) === String(ChannelType.GuildVoice) ||
        isThreadLike(candidate as { type?: ChannelType | number | string });

    return Boolean(
        isSupportedType &&
            candidate.id &&
            candidate.name &&
            candidate.messages &&
            typeof candidate.messages.fetch === "function"
    );
}

function getParentCategoryMeta(channel: CrawlableChannel): {
    parentCategoryId: string | null;
    parentCategoryName: string | null;
} {
    if (isThreadLike(channel)) {
        return {
            parentCategoryId: channel.parent?.parent?.id || null,
            parentCategoryName: channel.parent?.parent?.name || null,
        };
    }

    return {
        parentCategoryId: channel.parent?.id || null,
        parentCategoryName: channel.parent?.name || null,
    };
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
            .filter((channel) => isCrawlableChannel(channel))
            .map((channel) => channel as CrawlableChannel)
            .sort((left, right) => left.name.localeCompare(right.name));
    }

    public static async rankCandidateChannels(
        guild: Guild | null,
        query: string,
        currentChannelId?: string | null
    ): Promise<Array<{
        channelId: string;
        channelName: string;
        guildId: string | null;
        isIndexed: boolean;
        matchSource: "memory" | "live_name";
        lastIndexedTimestamp: number | null;
    }>> {
        const normalizedTerms = query
            .toLowerCase()
            .split(/\s+/)
            .map((term) => term.trim())
            .filter((term) => term.length > 1);

        const knownChannels = new Map(
            (await DiscordMemoryService.getKnownChannelsAsync(guild?.id || null)).map((channel) => [
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
        limit = getInteractiveCrawlLimit(),
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
                oldestFetchedMessageId: null,
                queryHint: queryHint || null,
                backgroundIngestQueued: false,
                previewMessages: [],
            };
        }

        const channel = guild.channels.cache.get(channelId);
        if (!isCrawlableChannel(channel)) {
            return {
                channelId,
                channelName: channel?.name || channelId,
                messagesFetched: 0,
                messagesStored: 0,
                exhausted: true,
                oldestFetchedMessageId: null,
                queryHint: queryHint || null,
                backgroundIngestQueued: false,
                previewMessages: [],
            };
        }

        const parentMeta = getParentCategoryMeta(channel);
        await DiscordMemoryService.upsertDiscoveredChannel(
            channel.id,
            guild.id,
            channel.name,
            Date.now(),
            {
                channelType: String(channel.type),
                parentCategoryId: parentMeta.parentCategoryId,
                parentCategoryName: parentMeta.parentCategoryName,
            }
        );

        const crawlState = (await DiscordMemoryService.getChannelCrawlStateAsync(channel.id))[0] || null;
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
                oldestFetchedMessageId: crawlState?.oldestFetchedMessageId || null,
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

        await DiscordMemoryService.updateChannelCrawlState(channel.id, before || null, exhausted);
        await enrichMessagesWithGuildMembers(guild, fetchedMessages);
        const previewMessages = buildPreviewMessages(fetchedMessages, queryHint);
        this.enqueueBackgroundIngest(fetchedMessages);

        return {
            channelId: channel.id,
            channelName: channel.name,
            messagesFetched: fetched,
            messagesStored: stored,
            exhausted,
            oldestFetchedMessageId: before || null,
            queryHint: queryHint || null,
            backgroundIngestQueued: stored > 0,
            previewMessages,
        };
    }

    public static async waitForBackgroundIngest(): Promise<void> {
        await backgroundIngestQueue;
    }

    /**
     * One-shot crawl targeting a specific time range using a Discord snowflake
     * computed from the timestamp. Does NOT update channel crawl state.
     */
    public static async crawlChannelMessagesAtTime(
        guild: Guild | null,
        channelId: string,
        beforeTimestamp: number,
        limit: number,
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
                oldestFetchedMessageId: null,
                queryHint: queryHint || null,
                backgroundIngestQueued: false,
                previewMessages: [],
            };
        }

        const channel = guild.channels.cache.get(channelId);
        if (!isCrawlableChannel(channel)) {
            return {
                channelId,
                channelName: channel?.name || channelId,
                messagesFetched: 0,
                messagesStored: 0,
                exhausted: true,
                oldestFetchedMessageId: null,
                queryHint: queryHint || null,
                backgroundIngestQueued: false,
                previewMessages: [],
            };
        }

        const DISCORD_EPOCH = 1420070400000;
        const DISCORD_SNOWFLAKE_INCREMENT = BigInt(4194304);
        const snowflakeNum =
            (BigInt(beforeTimestamp) - BigInt(DISCORD_EPOCH)) * DISCORD_SNOWFLAKE_INCREMENT;
        let before: string | undefined = snowflakeNum.toString();
        let fetched = 0;
        let exhausted = false;
        const fetchedMessages: Array<Message> = [];

        while (fetched < limit) {
            const batch: Collection<string, Message> = await channel.messages.fetch({
                limit: Math.min(100, limit - fetched),
                before,
            });

            if (!batch.size) {
                exhausted = true;
                break;
            }

            const messages: Message[] = [...batch.values()].sort(
                (left, right) => left.createdTimestamp - right.createdTimestamp
            );
            fetchedMessages.push(...messages);

            fetched += batch.size;
            before = messages[0]?.id;
            await onProgress?.(
                "crawl_channel_messages_at_time",
                `coletadas ${fetched}/${limit} mensagens de ${channel.name} (targeted)`
            );
        }

        await enrichMessagesWithGuildMembers(guild, fetchedMessages);
        const previewMessages = buildPreviewMessages(fetchedMessages, queryHint);
        this.enqueueBackgroundIngest(fetchedMessages);

        return {
            channelId: channel.id,
            channelName: channel.name,
            messagesFetched: fetched,
            messagesStored: fetchedMessages.length,
            exhausted,
            oldestFetchedMessageId: before || null,
            queryHint: queryHint || null,
            backgroundIngestQueued: fetchedMessages.length > 0,
            previewMessages,
        };
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

    /** Extract searchable content from a message, including embeds. */
    function getEffectiveContent(message: any): string {
        const parts: string[] = [];
        if (message.content) parts.push(String(message.content));
        if (message.embeds?.length) {
            for (const embed of message.embeds) {
                if (embed.title) parts.push(String(embed.title));
                if (embed.description) parts.push(String(embed.description));
            }
        }
        return parts.join(" ");
    }

    const candidates = messages
        .filter((message) => Boolean(getEffectiveContent(message).trim()))
        .map((message) => {
            const effectiveContent = getEffectiveContent(message);
            const normalizedContent = normalizeLookupValue(effectiveContent);
            const score = normalizedTerms.reduce((total, term) => {
                return total + (normalizedContent.includes(term) ? 1 : 0);
            }, 0);

            return {
                message,
                effectiveContent,
                score,
            };
        });

    const ranked = candidates
        .filter((item) => item.score > 0 || normalizedTerms.length === 0)
        .sort((left, right) => {
            if (right.score !== left.score) {
                return right.score - left.score;
            }

            return (right.message.createdTimestamp || 0) - (left.message.createdTimestamp || 0);
        })
        .slice(0, PREVIEW_MESSAGE_LIMIT);

    const effective = ranked.length
        ? ranked
        : candidates
              .sort(
                  (left, right) =>
                      (right.message.createdTimestamp || 0) - (left.message.createdTimestamp || 0)
              )
              .slice(0, PREVIEW_MESSAGE_LIMIT);

    return effective.map(({ message, effectiveContent }) => {
        const authorIdentity = resolveMessageAuthorIdentity(message);
        return {
        messageId: String(message.id),
        authorId: String(message.author?.id || ""),
        authorName: authorIdentity.authorName,
        authorUsername: authorIdentity.authorUsername,
        authorNickname: authorIdentity.authorNickname,
        content: effectiveContent || String(message.content || ""),
        createdTimestamp: Number(message.createdTimestamp || 0),
        jumpLink:
            typeof message.url === "string"
                ? message.url
                : `https://discord.com/channels/${message.guildId}/${message.channelId}/${message.id}`,
        };
    });
}

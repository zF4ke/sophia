import { z } from "zod";
import { Routes } from "discord.js";
import { T } from "@/shared/discordTools";
import { withDiscordRateLimitRetry } from "@/discord/live/discordRateLimitRetry";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { StoredMessage } from "@/memory/types";
import type { EvidenceItem } from "@/runtime/contracts";
import type { DiscordToolResult } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

// ── Types ──────────────────────────────────────────────────────────

interface DiscordSearchMessage {
    id: string;
    channel_id: string;
    author: {
        id: string;
        username: string;
        global_name?: string | null;
        bot?: boolean;
    };
    content: string;
    timestamp: string;
    pinned?: boolean;
    attachments?: Array<{ url: string; filename: string }>;
    embeds?: unknown[];
    message_reference?: { message_id?: string } | null;
    member?: {
        nick?: string | null;
    };
    hit?: boolean;
}

interface DiscordSearchResponse {
    messages: DiscordSearchMessage[][];
    total_results: number;
    analytics_id?: string;
}

export interface SearchMessagesResult {
    query: string | null;
    totalResults: number;
    returnedCount: number;
    messages: SearchResultMessage[];
    hasMore: boolean;
    nextOffset: number;
    filters: Record<string, unknown>;
}

interface SearchResultMessage {
    messageId: string;
    channelId: string;
    channelName: string;
    channelTopic: string | null;
    authorId: string;
    authorName: string;
    authorUsername: string;
    content: string;
    createdTimestamp: number;
    jumpLink: string;
    pinned: boolean;
    isBot: boolean;
    attachments: string[];
}

// ── JSON Schema parameters ─────────────────────────────────────────

const parameters = {
    type: "object",
    properties: {
        content: {
            type: "string",
            description: "Text content to search for. Works like Discord's search bar.",
        },
        author_id: {
            type: "string",
            description:
                "Filter to messages from this user ID. Resolve the user first with resolve_member_identity.",
        },
        channel_id: {
            type: "string",
            description: "Filter to messages in this channel ID.",
        },
        mentions: {
            type: "string",
            description: "Filter to messages that mention this user ID.",
        },
        has: {
            type: "string",
            description:
                "Filter by attachment type. Valid values: link, embed, file, video, image, sound, sticker.",
        },
        pinned: {
            type: "string",
            description: "Filter by pinned status. Valid values: true, false.",
        },
        author_type: {
            type: "string",
            description:
                "Filter by author type. Valid values: user, bot, webhook.",
        },
        before: {
            type: "string",
            description:
                "Only messages before this snowflake ID. Use for pagination: pass the ID of the last message from the previous page.",
        },
        after: {
            type: "string",
            description: "Only messages after this snowflake ID.",
        },
        sort_by: {
            type: "string",
            description:
                "Sort results. Valid values: timestamp, relevance. Default: relevance.",
        },
        sort_order: {
            type: "string",
            description:
                "Sort direction. Valid values: asc, desc. Default: desc.",
        },
        offset: {
            type: "number",
            description:
                "Numeric offset for pagination (max 5000). Prefer using before/after snowflake IDs for large result sets.",
        },
        limit: {
            type: "number",
            description:
                "Number of results per page (max 25, default 25).",
        },
    },
    required: [],
} as const;

// ── Helpers ────────────────────────────────────────────────────────

function buildQueryParams(args: Record<string, unknown>): URLSearchParams {
    const params = new URLSearchParams();

    const stringFields = [
        "content",
        "author_id",
        "channel_id",
        "mentions",
        "has",
        "pinned",
        "author_type",
        "sort_by",
        "sort_order",
    ] as const;

    for (const field of stringFields) {
        const value = args[field];
        if (value != null && String(value).trim()) {
            params.set(field, String(value).trim());
        }
    }

    // min_id / max_id mapped from before / after
    if (args.before != null && String(args.before).trim()) {
        params.set("max_id", String(args.before).trim());
    }
    if (args.after != null && String(args.after).trim()) {
        params.set("min_id", String(args.after).trim());
    }

    if (args.offset != null) {
        params.set("offset", String(Math.min(5000, Math.max(0, Number(args.offset)))));
    }

    // Discord API caps at 25 per search page
    const limit = args.limit != null ? Math.min(25, Math.max(1, Number(args.limit))) : 25;
    params.set("limit", String(limit));

    return params;
}

function extractHitMessages(response: DiscordSearchResponse): DiscordSearchMessage[] {
    const hits: DiscordSearchMessage[] = [];
    for (const group of response.messages) {
        for (const msg of group) {
            if (msg.hit) {
                hits.push(msg);
            }
        }
    }
    return hits;
}

function toSearchResultMessage(
    msg: DiscordSearchMessage,
    guildId: string,
    channelNames: Map<string, string>,
    channelTopics: Map<string, string | null>,
): SearchResultMessage {
    const authorName =
        msg.member?.nick || msg.author.global_name || msg.author.username;
    return {
        messageId: msg.id,
        channelId: msg.channel_id,
        channelName: channelNames.get(msg.channel_id) || msg.channel_id,
        channelTopic: channelTopics.get(msg.channel_id) || null,
        authorId: msg.author.id,
        authorName,
        authorUsername: msg.author.username,
        content: msg.content,
        createdTimestamp: new Date(msg.timestamp).getTime(),
        jumpLink: `https://discord.com/channels/${guildId}/${msg.channel_id}/${msg.id}`,
        pinned: Boolean(msg.pinned),
        isBot: Boolean(msg.author.bot),
        attachments: (msg.attachments || []).map((a) => a.url),
    };
}

function toStoredMessage(
    msg: DiscordSearchMessage,
    guildId: string,
    channelNames: Map<string, string>,
): StoredMessage {
    const authorName =
        msg.member?.nick || msg.author.global_name || msg.author.username;
    return {
        id: msg.id,
        guildId,
        channelId: msg.channel_id,
        channelName: channelNames.get(msg.channel_id) || msg.channel_id,
        authorId: msg.author.id,
        authorName,
        authorUsername: msg.author.username,
        authorNickname: msg.member?.nick ?? null,
        content: msg.content,
        attachmentsJson: JSON.stringify(msg.attachments || []),
        referenceMessageId: msg.message_reference?.message_id ?? null,
        createdTimestamp: new Date(msg.timestamp).getTime(),
        jumpLink: `https://discord.com/channels/${guildId}/${msg.channel_id}/${msg.id}`,
        isBot: msg.author.bot ? 1 : 0,
    };
}

// ── Tool definition ────────────────────────────────────────────────

export const searchMessagesTool: ToolDefinition = {
    name: T.search_messages,

    catalog: {
        effect: "read",
        description:
            "Search guild messages using Discord's native search API with full filter support.",
        evidenceRole: "message_evidence",
    },

    schema: {
        description:
            "Search messages using Discord's native search engine. Supports all Discord search filters: content text, author, channel, mentions, attachment type (link/embed/file/video/image/sound/sticker), pinned status, author type (user/bot/webhook), and date ranges via snowflake IDs. Use for narrow, specific searches. Results are paginated (max 25 per page) — use the 'before' parameter with the last message ID to get the next page. Returned messages are sorted chronologically (oldest first) within each page.",
        parameters,
    },

    capability: {
        description:
            "Search guild messages via Discord's native search. Exposes all Discord search filters: content, author_id, channel_id, mentions, has (link/embed/file/video/image/sound/sticker), pinned, author_type (user/bot/webhook), date bounds via snowflake IDs. Use for precise, filtered searches that retrieve_messages can't handle efficiently.",
        inputSchema: z.object({
            content: z.string().optional().describe("Text content to search for."),
            author_id: z.string().optional().describe("Filter by author user ID."),
            channel_id: z.string().optional().describe("Filter by channel ID."),
            mentions: z.string().optional().describe("Filter by mentioned user ID."),
            has: z.string().optional().describe("Filter by attachment type: link, embed, file, video, image, sound, sticker."),
            pinned: z.string().optional().describe("Filter pinned messages: true or false."),
            author_type: z.string().optional().describe("Filter by author type: user, bot, webhook."),
            before: z.string().optional().describe("Snowflake ID for pagination — messages before this ID."),
            after: z.string().optional().describe("Snowflake ID — messages after this ID."),
            sort_by: z.string().optional().describe("Sort by: timestamp or relevance."),
            sort_order: z.string().optional().describe("Sort direction: asc or desc."),
            offset: z.number().optional().describe("Numeric offset for pagination (max 5000)."),
            limit: z.number().optional().describe("Results per page (max 25)."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context must exist"],
        postconditions: ["returns Discord search results as message evidence"],

        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.search_messages,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }

            const guild = context.guild;
            const queryParams = buildQueryParams(args as Record<string, unknown>);

            let response: DiscordSearchResponse;
            try {
                response = (await withDiscordRateLimitRetry(() =>
                    guild.client.rest.get(Routes.guildMessagesSearch(guild.id), {
                        query: queryParams,
                    }),
                )) as DiscordSearchResponse;
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                return {
                    tool: T.search_messages,
                    summary: `Search failed: ${message}`,
                    data: null,
                    errorMessage: message,
                };
            }

            const hits = extractHitMessages(response);

            // Build channel name map from guild cache
            const channelNames = new Map<string, string>();
            const channelTopics = new Map<string, string | null>();
            for (const msg of hits) {
                if (!channelNames.has(msg.channel_id)) {
                    const channel = guild.channels.cache.get(msg.channel_id);
                    channelNames.set(msg.channel_id, channel?.name || msg.channel_id);
                    if (channel && "topic" in channel) {
                        const topic = (channel as typeof channel & { topic?: unknown }).topic;
                        channelTopics.set(
                            msg.channel_id,
                            typeof topic === "string" && topic.trim() ? topic.trim() : null,
                        );
                    } else {
                        channelTopics.set(msg.channel_id, null);
                    }
                }
            }

            // Sort chronologically (oldest first)
            hits.sort(
                (a, b) =>
                    new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
            );

            const messages = hits.map((msg) =>
                toSearchResultMessage(msg, guild.id, channelNames, channelTopics),
            );

            // Best-effort background ingestion — fire-and-forget
            const storedMessages = hits.map((msg) =>
                toStoredMessage(msg, guild.id, channelNames),
            );
            if (storedMessages.length > 0) {
                void Promise.all(
                    storedMessages.map((sm) =>
                        DiscordMemoryService.ingestStoredMessage(sm).catch(() => {}),
                    ),
                );
            }

            const requestedLimit = queryParams.get("limit")
                ? Number(queryParams.get("limit"))
                : 25;
            const currentOffset = queryParams.get("offset")
                ? Number(queryParams.get("offset"))
                : 0;
            const hasMore =
                currentOffset + messages.length < response.total_results;

            const result: SearchMessagesResult = {
                query: String(args.content || "") || null,
                totalResults: response.total_results,
                returnedCount: messages.length,
                messages,
                hasMore,
                nextOffset: currentOffset + messages.length,
                filters: Object.fromEntries(
                    Object.entries(args as Record<string, unknown>).filter(
                        ([k, v]) =>
                            v != null &&
                            !["limit", "offset", "before", "after"].includes(k),
                    ),
                ),
            };

            const summaryParts = [
                `${response.total_results} total results`,
                `showing ${messages.length}`,
            ];
            if (hasMore) summaryParts.push("more available");

            return {
                tool: T.search_messages,
                summary: summaryParts.join(", "),
                data: result,
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult): EvidenceItem[] {
            if (!run.data) return [];
            const result = run.data as SearchMessagesResult;
            if (!result.messages?.length) return [];

            return result.messages.map((msg) => ({
                tool: T.search_messages,
                summary: run.summary,
                content: `[${msg.authorName} in #${msg.channelName}${msg.channelTopic ? ` (topic: ${msg.channelTopic.replace(/\s+/g, " ").trim().slice(0, 100)})` : ""}] ${msg.content}`,
                evidenceRole: "message_evidence" as const,
                strength:
                    msg.content.length > 30
                        ? ("strong" as const)
                        : ("weak" as const),
                sourceOrigin: "live_refresh" as const,
                messageId: msg.messageId,
                authorId: msg.authorId,
                authorName: msg.authorName,
                authorUsername: msg.authorUsername,
                channelId: msg.channelId,
                channelName: msg.channelName,
                jumpLink: msg.jumpLink,
                createdTimestamp: msg.createdTimestamp,
            }));
        },
    },

    display: { icon: "🔎", labelPt: "Pesquisa Discord" },
};

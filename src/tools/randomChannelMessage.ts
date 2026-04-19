import { z } from "zod";
import { T } from "@/shared/discordTools";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { EvidenceItem } from "@/runtime/contracts";
import type { DiscordToolResult } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

interface RandomChannelMessageResult {
    messageId: string;
    channelId: string;
    channelMention: string;
    channelName: string;
    authorId: string;
    authorName: string;
    authorUsername: string | null;
    content: string;
    createdTimestamp: number;
    jumpLink: string;
    referenceMessageId: string | null;
}

const MAX_COUNT = 25;

const parameters = {
    type: "object",
    properties: {
        channel_id: {
            type: "string",
            description: "Channel ID to sample from. Messages are chosen from already ingested local messages for this channel.",
        },
        count: {
            type: "number",
            description: `Optional number of distinct random messages to return (1-${MAX_COUNT}). Defaults to 1. Use this to fulfill "give me N random messages" requests in a single call instead of calling the tool repeatedly.`,
        },
        author_id: {
            type: "string",
            description: "Optional author user ID filter. When set, sample only from this author's ingested messages in the channel.",
        },
        after_timestamp: {
            type: "number",
            description: "Optional lower timestamp bound in unix milliseconds (inclusive).",
        },
        before_timestamp: {
            type: "number",
            description: "Optional upper timestamp bound in unix milliseconds (exclusive).",
        },
    },
    required: ["channel_id"],
} as const;

function toResult(stored: {
    id: string;
    channelId: string;
    channelName: string;
    authorId: string;
    authorName: string;
    authorUsername?: string | null;
    content: string;
    createdTimestamp: number;
    jumpLink: string;
    referenceMessageId: string | null;
}): RandomChannelMessageResult {
    return {
        messageId: stored.id,
        channelId: stored.channelId,
        channelMention: `<#${stored.channelId}>`,
        channelName: stored.channelName,
        authorId: stored.authorId,
        authorName: stored.authorName,
        authorUsername: stored.authorUsername ?? null,
        content: stored.content,
        createdTimestamp: stored.createdTimestamp,
        jumpLink: stored.jumpLink,
        referenceMessageId: stored.referenceMessageId,
    };
}

function toEvidence(message: RandomChannelMessageResult, summary: string): EvidenceItem {
    return {
        tool: T.random_channel_message,
        summary,
        content: `[${message.authorName} in #${message.channelName}] ${message.content}`,
        evidenceRole: "message_evidence",
        strength: message.content.length > 30 ? "strong" : "weak",
        sourceOrigin: "cache",
        messageId: message.messageId,
        authorId: message.authorId,
        authorName: message.authorName,
        authorUsername: message.authorUsername,
        channelId: message.channelId,
        channelName: message.channelName,
        jumpLink: message.jumpLink,
        createdTimestamp: message.createdTimestamp,
    };
}

export const randomChannelMessageTool: ToolDefinition = {
    name: T.random_channel_message,

    catalog: {
        effect: "read",
        description: "Pick one or more random ingested messages from a channel, optionally filtered by author or time bounds.",
        evidenceRole: "message_evidence",
    },

    schema: {
        description:
            "Pick one or more random messages from the local ingested message cache for a channel. Pass `count` (1-25) to sample multiple distinct messages in a single call — use this when the user asks for N random messages from a chat instead of calling the tool repeatedly.",
        parameters,
    },

    capability: {
        description:
            "Sample random ingested messages from a channel, with optional author and timestamp filters. Supports returning up to 25 messages per call via `count`.",
        inputSchema: z.object({
            channel_id: z.string().describe("Channel ID to sample from."),
            count: z.number().optional().describe("Number of distinct random messages to return (1-25). Defaults to 1."),
            author_id: z.string().optional().describe("Optional author user ID filter."),
            after_timestamp: z
                .number()
                .optional()
                .describe("Optional lower timestamp bound in unix milliseconds (inclusive)."),
            before_timestamp: z
                .number()
                .optional()
                .describe("Optional upper timestamp bound in unix milliseconds (exclusive)."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: ["channel messages must already be ingested locally"],
        postconditions: ["returns up to `count` random ingested messages or reports that no ingested message matched"],
        async run(context, args) {
            const channelId = String(args.channel_id || "").trim();
            if (!channelId) {
                return {
                    tool: T.random_channel_message,
                    summary: "No channel ID provided.",
                    data: null,
                    errorMessage: "channel_id is required.",
                };
            }

            const guildId = context.guild?.id ?? null;
            const authorId =
                args.author_id != null && String(args.author_id).trim()
                    ? String(args.author_id).trim()
                    : null;
            const afterTimestamp =
                typeof args.after_timestamp === "number" && Number.isFinite(args.after_timestamp)
                    ? Number(args.after_timestamp)
                    : null;
            const beforeTimestamp =
                typeof args.before_timestamp === "number" && Number.isFinite(args.before_timestamp)
                    ? Number(args.before_timestamp)
                    : null;

            const rawCount =
                typeof args.count === "number" && Number.isFinite(args.count)
                    ? Math.floor(Number(args.count))
                    : 1;
            const count = Math.max(1, Math.min(MAX_COUNT, rawCount));
            const isBatch = args.count != null && count > 1;

            const stored = await DiscordMemoryService.getRandomStoredMessagesAsync(
                {
                    guildId,
                    channelIds: [channelId],
                    authorIds: authorId ? [authorId] : undefined,
                    afterTimestamp,
                    beforeTimestamp,
                },
                count,
            );

            if (!stored.length) {
                const filterBits = [
                    authorId ? `author ${authorId}` : null,
                    afterTimestamp != null ? `after ${afterTimestamp}` : null,
                    beforeTimestamp != null ? `before ${beforeTimestamp}` : null,
                ].filter(Boolean);
                return {
                    tool: T.random_channel_message,
                    summary: `No ingested messages matched in <#${channelId}>${filterBits.length ? ` (${filterBits.join(", ")})` : ""}.`,
                    data: null,
                };
            }

            const results = stored.map(toResult);
            const channelName = results[0].channelName;

            if (!isBatch) {
                const single = results[0];
                return {
                    tool: T.random_channel_message,
                    summary: `Random ingested message from #${channelName} (${single.messageId}).`,
                    data: single,
                };
            }

            return {
                tool: T.random_channel_message,
                summary: `Sampled ${results.length} random ingested message(s) from #${channelName}.`,
                data: { messages: results, count: results.length },
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult): EvidenceItem[] {
            if (!run.data) return [];
            const data = run.data as
                | RandomChannelMessageResult
                | { messages: RandomChannelMessageResult[]; count: number };
            if (Array.isArray((data as { messages?: unknown[] }).messages)) {
                const batch = data as { messages: RandomChannelMessageResult[] };
                return batch.messages.map((m) => toEvidence(m, run.summary));
            }
            return [toEvidence(data as RandomChannelMessageResult, run.summary)];
        },
    },

    display: { icon: "🎲", labelPt: "Mensagem aleatória" },
};

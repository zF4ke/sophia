import { z } from "zod";
import type { DiscordToolName } from "@/shared/discordTools";

/**
 * JSON-Schema-compatible tool definitions for native function calling.
 * Each tool maps directly to a CapabilityRegistry capability.
 */

export interface NativeToolDef {
    type: "function";
    function: {
        name: string;
        description: string;
        parameters: Record<string, unknown>;
    };
}

const retrieveMessagesParams = {
    type: "object",
    properties: {
        query: {
            type: "string",
            description: "Search query for Discord messages. Used for both lexical and semantic matching.",
        },
        channelIds: {
            type: "array",
            items: { type: "string" },
            description: "Limit search to these channel IDs (from resolve_channel_targets or list_guild_structure). Omit to search all indexed channels.",
        },
        authorId: {
            type: "string",
            description: "Filter messages to this author's Discord snowflake ID (from resolve_member_identity).",
        },
        beforeTimestamp: {
            type: "number",
            description: "Unix timestamp (ms). Only return messages before this time. When computing a year for a partial date (day+month only): if that day/month has already passed relative to `current_date`, use the current year; if it is still upcoming, use the previous year.",
        },
        afterTimestamp: {
            type: "number",
            description: "Unix timestamp (ms). Only return messages after this time. When computing a year for a partial date (day+month only): if that day/month has already passed relative to `current_date`, use the current year; if it is still upcoming, use the previous year.",
        },
        aroundMessageId: {
            type: "string",
            description: "A Discord message ID from previous results. Returns messages surrounding that message for reading context. Use this to zoom into a specific conversation point.",
        },
        mode: {
            type: "string",
            enum: ["history", "semantic", "mixed"],
            description: "Retrieval mode. 'history' = chronological order (best for reading through a channel like a file). 'semantic' = relevance-ranked search. 'mixed' = both lanes. Default: mixed.",
        },
        limit: {
            type: "number",
            description: "Number of messages to return per page. Default 50. Use higher values when scrolling through history.",
        },
        cursor: {
            type: "object",
            description: "Continuation cursor from a previous retrieve_messages call. Pass this to fetch the next page and keep scrolling through history.",
            properties: {
                history: {
                    type: "object",
                    additionalProperties: { type: ["string", "null"] },
                },
                semantic: {
                    type: "object",
                    properties: {
                        lastScore: { type: "number" },
                        lastCreatedTimestamp: { type: "number" },
                        lastMessageId: { type: "string" },
                    },
                    required: ["lastScore", "lastCreatedTimestamp", "lastMessageId"],
                },
            },
        },
        excludedMessageIds: {
            type: "array",
            items: { type: "string" },
            description: "Message IDs to exclude from results (already seen in previous pages).",
        },
    },
    required: ["query"],
} as const;

const resolveMemberIdentityParams = {
    type: "object",
    properties: {
        query: {
            type: "string",
            description: "Member name, username, nickname, or Discord snowflake ID.",
        },
    },
    required: ["query"],
} as const;

const listGuildStructureParams = {
    type: "object",
    properties: {
        targetText: {
            type: "string",
            description: "Optional text to focus on specific channels/categories.",
        },
    },
} as const;

const resolveChannelTargetsParams = {
    type: "object",
    properties: {
        targetText: {
            type: "string",
            description: "Channel or category name, ID, or mention to resolve.",
        },
    },
    required: ["targetText"],
} as const;

const getMemberProfileParams = {
    type: "object",
    properties: {
        nameOrId: {
            type: "string",
            description: "Member name, username, nickname, or Discord snowflake ID.",
        },
    },
    required: ["nameOrId"],
} as const;

const listMembersParams = {
    type: "object",
    properties: {
        filters: {
            type: "string",
            description: "Optional name or username fragment to filter by.",
        },
        limit: {
            type: "number",
            description: "Page size. Default 20.",
        },
        offset: {
            type: "number",
            description: "Pagination offset.",
        },
    },
} as const;

const getGuildContextParams = {
    type: "object",
    properties: {},
} as const;

const finishParams = {
    type: "object",
    properties: {
        answer: {
            type: "string",
            description: "Your final answer to the user. This will be sent as Sophia's response in Discord.",
        },
    },
    required: ["answer"],
} as const;

export const TOOL_DEFINITIONS: NativeToolDef[] = [
    {
        type: "function",
        function: {
            name: "retrieve_messages",
            description:
                "Search Discord message history. Returns messages with full metadata including message IDs, author IDs, channel IDs, timestamps, and jump links. Use 'mode' to control retrieval: 'history' for chronological scrolling (like reading a file), 'semantic' for relevance-based search, 'mixed' for both. Use 'cursor' from previous results to paginate through more messages. Use 'aroundMessageId' to get context surrounding a specific message ID.",
            parameters: retrieveMessagesParams,
        },
    },
    {
        type: "function",
        function: {
            name: "resolve_member_identity",
            description:
                "Resolve a member or bot in the current guild. Pass a Discord snowflake ID, username, display name, or nickname. Returns the member's resolved identity with their Discord ID, username, display name, roles, and guild membership status. Use the returned ID for filtering in other tools.",
            parameters: resolveMemberIdentityParams,
        },
    },
    {
        type: "function",
        function: {
            name: "list_guild_structure",
            description:
                "List channels and categories in the current guild with their IDs and types. Use this to discover which channels exist before searching them. Returns channel IDs you can pass to retrieve_messages.",
            parameters: listGuildStructureParams,
        },
    },
    {
        type: "function",
        function: {
            name: "resolve_channel_targets",
            description:
                "Resolve channel or category references in the current guild. Pass a channel name, ID, or mention. If a category is matched, expands it to all child message channels. Returns resolved channel IDs you can use with retrieve_messages.",
            parameters: resolveChannelTargetsParams,
        },
    },
    {
        type: "function",
        function: {
            name: "get_member_profile",
            description:
                "Fetch a detailed live guild member profile. Returns roles, join date, account age, nickname, bot status, Nitro/premium status, avatar URL, and Discord ID. Use the returned ID for filtering messages by author.",
            parameters: getMemberProfileParams,
        },
    },
    {
        type: "function",
        function: {
            name: "list_members",
            description:
                "List live guild members sorted by join date with their IDs, usernames, display names, and roles. Supports pagination with offset and optional name/username fragment filter.",
            parameters: listMembersParams,
        },
    },
    {
        type: "function",
        function: {
            name: "get_guild_context",
            description:
                "Fetch live guild metadata: name, member count, channel count, creation date, icon, and guild ID.",
            parameters: getGuildContextParams,
        },
    },
    {
        type: "function",
        function: {
            name: "finish",
            description:
                "Call this when you have your final answer ready. The 'answer' field will be sent as Sophia's response in Discord. You MUST call this tool to deliver your response — do not just output text.",
            parameters: finishParams,
        },
    },
];

/** Map of tool name → NativeToolDef for quick lookup. */
export const TOOL_BY_NAME = Object.fromEntries(
    TOOL_DEFINITIONS.map((def) => [def.function.name, def])
) as Record<string, NativeToolDef>;

/** All valid tool names including 'finish'. */
export type NativeToolName = DiscordToolName | "finish";

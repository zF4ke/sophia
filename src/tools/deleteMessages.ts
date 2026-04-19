import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        channel_id: {
            type: "string",
            description:
                "The channel ID where the messages live (from resolve_channel_targets or list_guild_structure).",
        },
        message_ids: {
            type: "array",
            items: { type: "string" },
            description:
                "The Discord snowflake IDs of the specific messages to delete (1-50). Obtain these by searching first (e.g. search_messages or retrieve_messages in the target channel) — never guess or fabricate IDs.",
        },
    },
    required: ["channel_id", "message_ids"],
} as const;

export const deleteMessagesTool: ToolDefinition = {
    name: T.delete_messages,

    catalog: {
        effect: "destructive",
        description:
            "Delete one or more specific messages by ID from a Discord channel. Destructive — requires admin approval with confirmation modal.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Delete one or more specific messages from a channel by their IDs. This is a DESTRUCTIVE action that requires admin approval. Use only when the user explicitly asks to delete specific message(s). IMPORTANT: You must obtain the real message IDs first — use search_messages or retrieve_messages to locate the target messages, then pass their IDs here. Never guess or fabricate message IDs. For bulk deletion of the most recent history without specific targets, use clear_messages instead.",
        parameters,
    },

    capability: {
        description:
            "Delete specific messages by ID from a Discord channel. Destructive — requires admin approval with confirmation modal.",
        inputSchema: z.object({
            channel_id: z.string().describe("Channel ID where the messages are."),
            message_ids: z
                .array(z.string())
                .min(1)
                .max(50)
                .describe("IDs of the messages to delete (1-50)."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "destructive",
        authRequirements: ["admin"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: [
            "guild context should exist",
            "channel must be accessible",
            "message IDs must be known",
        ],
        postconditions: [
            "deletes the specified messages from the target channel",
        ],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.delete_messages,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const channel = context.guild.channels.cache.get(
                String(args.channel_id),
            );
            if (!channel || !channel.isTextBased()) {
                return {
                    tool: T.delete_messages,
                    summary: "Channel not found or not text-based.",
                    data: null,
                    errorMessage: "Channel not found or not text-based.",
                };
            }
            const rawIds = Array.isArray(args.message_ids)
                ? (args.message_ids as unknown[])
                : [];
            const messageIds = Array.from(
                new Set(
                    rawIds
                        .map((id) => String(id).trim())
                        .filter((id) => id.length > 0),
                ),
            ).slice(0, 50);
            if (messageIds.length === 0) {
                return {
                    tool: T.delete_messages,
                    summary: "No message IDs provided.",
                    data: { deletedCount: 0, deletedIds: [], failedIds: [] },
                    errorMessage: "No message IDs provided.",
                };
            }

            const deletedIds: string[] = [];
            const failedIds: Array<{ id: string; reason: string }> = [];

            for (const id of messageIds) {
                try {
                    const message = await channel.messages.fetch(id);
                    await message.delete();
                    deletedIds.push(id);
                } catch (err) {
                    failedIds.push({
                        id,
                        reason:
                            err instanceof Error ? err.message : "unknown error",
                    });
                }
            }

            const channelMention = `<#${channel.id}>`;
            const summary =
                failedIds.length === 0
                    ? `Deleted ${deletedIds.length} message(s) from ${channelMention}.`
                    : `Deleted ${deletedIds.length} of ${messageIds.length} message(s) from ${channelMention} (${failedIds.length} failed).`;

            return {
                tool: T.delete_messages,
                summary,
                data: {
                    deletedCount: deletedIds.length,
                    deletedIds,
                    failedIds,
                    channelId: channel.id,
                    channelMention,
                    channelName: channel.name,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "🗑️", labelPt: "Apagar mensagens" },

    describeApproval(args: ToolArguments) {
        const ids = Array.isArray(args.message_ids)
            ? (args.message_ids as unknown[])
            : [];
        const count = ids.length;
        return `Apagar ${count} mensagem(ns) específica(s) em <#${args.channel_id ?? "desconhecido"}>`;
    },
};

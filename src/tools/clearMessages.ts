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
                "The channel ID to delete messages from (from resolve_channel_targets or list_guild_structure).",
        },
        count: {
            type: "number",
            description: "Number of messages to delete (1-100).",
        },
        author_id: {
            type: "string",
            description:
                "Optional: only delete messages from this author (Discord snowflake ID).",
        },
    },
    required: ["channel_id", "count"],
} as const;

export const clearMessagesTool: ToolDefinition = {
    name: T.clear_messages,

    catalog: {
        effect: "destructive",
        description:
            "Delete messages from a Discord channel. Destructive — follows the requester action tier and approval mode with confirmation modal.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Delete messages from a channel. This is a DESTRUCTIVE action that follows the requester action tier and approval mode before execution. Use only when explicitly asked to delete messages.",
        parameters,
    },

    capability: {
        description:
            "Delete messages from a Discord channel. Destructive — follows the requester action tier and approval mode with confirmation modal.",
        inputSchema: z.object({
            channel_id: z.string().describe("Channel ID to delete messages from."),
            count: z
                .number()
                .int()
                .min(1)
                .max(100)
                .describe("Number of messages to delete (1-100)."),
            author_id: z
                .string()
                .optional()
                .describe("Only delete messages from this author."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "destructive",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist", "channel must be accessible"],
        postconditions: ["deletes messages from the target channel"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.clear_messages,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const channel = context.guild.channels.cache.get(String(args.channel_id));
            if (!channel || !channel.isTextBased()) {
                return {
                    tool: T.clear_messages,
                    summary: "Channel not found or not text-based.",
                    data: null,
                    errorMessage: "Channel not found or not text-based.",
                };
            }
            const count = Math.min(Math.max(Number(args.count) || 1, 1), 100);
            let messages = await channel.messages.fetch({ limit: count });
            if (typeof args.author_id === "string" && args.author_id.trim()) {
                messages = messages.filter((m) => m.author.id === args.author_id);
            }
            if (messages.size === 0) {
                return {
                    tool: T.clear_messages,
                    summary: "No messages matched the criteria.",
                    data: { deletedCount: 0 },
                };
            }
            context.execution?.expectSourceDeletion([...messages.keys()]);
            const deleted = await (
                channel as import("discord.js").TextChannel
            ).bulkDelete(messages, true);
            const channelMention = `<#${channel.id}>`;
            return {
                tool: T.clear_messages,
                summary: `Deleted ${deleted.size} message(s) from ${channelMention}.`,
                data: {
                    deletedCount: deleted.size,
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

    display: { icon: "🗑️", labelPt: "Limpar mensagens" },

    describeApproval(args: ToolArguments) {
        return `Apagar ${args.count ?? "?"} mensagem(ns) em <#${args.channel_id ?? "desconhecido"}>`;
    },
};

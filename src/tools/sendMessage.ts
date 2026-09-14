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
                "The target channel or thread ID (from resolve_channel_targets, list_guild_structure, or list_threads).",
        },
        content: {
            type: "string",
            description: "The message content to send.",
        },
        reply_to: {
            type: "string",
            description: "Optional message ID to reply to.",
        },
    },
    required: ["channel_id", "content"],
} as const;

export const sendMessageTool: ToolDefinition = {
    name: T.send_message,
    publicationTarget: (_context, args) => String(args.channel_id),

    catalog: {
        effect: "write",
        description:
            "Send a message to a specific channel or thread. Write — follows the requester action tier and approval mode.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Send a message to a specific channel or thread. Write action — follows the requester action tier and approval mode. Use this when the user explicitly asks to send or post a message somewhere.",
        parameters,
    },

    capability: {
        description:
            "Send a message to a specific channel or thread. Write action — follows the requester action tier and approval mode.",
        inputSchema: z.object({
            channel_id: z
                .string()
                .describe("Target channel or thread ID."),
            content: z.string().describe("Message content to send."),
            reply_to: z
                .string()
                .optional()
                .describe("Message ID to reply to."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["sends a message in the target channel"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.send_message,
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
                    tool: T.send_message,
                    summary: "Channel not found or not text-based.",
                    data: null,
                    errorMessage: "Channel not found or not text-based.",
                };
            }
            const sendOptions: {
                content: string;
                reply?: { messageReference: string };
            } = {
                content: String(args.content),
            };
            if (
                typeof args.reply_to === "string" &&
                args.reply_to.trim()
            ) {
                sendOptions.reply = {
                    messageReference: args.reply_to.trim(),
                };
            }
            const sent = await channel.send(sendOptions);
            const channelMention = `<#${channel.id}>`;
            return {
                tool: T.send_message,
                summary: `Message sent in ${channelMention}.`,
                data: {
                    messageId: sent.id,
                    channelId: channel.id,
                    channelMention,
                    messageUrl: sent.url,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "✉️", labelPt: "Enviar mensagem" },

    describeApproval(args: ToolArguments) {
        return `Enviar mensagem em <#${args.channel_id ?? "desconhecido"}>`;
    },
};

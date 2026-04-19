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
                "The channel ID where the message is (from resolve_channel_targets or list_guild_structure).",
        },
        message_id: {
            type: "string",
            description:
                "The Discord snowflake ID of the message to edit. Obtain this by searching first (e.g. search_messages with author_type 'bot' in the target channel).",
        },
        content: {
            type: "string",
            description: "The new content for the message.",
        },
    },
    required: ["channel_id", "message_id", "content"],
} as const;

export const editMessageTool: ToolDefinition = {
    name: T.edit_message,

    catalog: {
        effect: "write",
        description:
            "Edit an existing message sent by the bot. Write — requires admin approval.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Edit an existing message that was sent by the bot. This is a WRITE action that requires admin approval. Use only when explicitly asked to edit a message. The bot can only edit its own messages. IMPORTANT: You must obtain the real message ID first — use search_messages (e.g. with author_type 'bot' in the target channel, or content keywords) to find the message, then pass its ID here. Never guess or fabricate a message ID.",
        parameters,
    },

    capability: {
        description:
            "Edit an existing message sent by the bot. Write — requires admin approval.",
        inputSchema: z.object({
            channel_id: z.string().describe("Channel ID where the message is."),
            message_id: z.string().describe("ID of the message to edit."),
            content: z.string().describe("New content for the message."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["admin"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: [
            "guild context should exist",
            "message must be authored by the bot",
        ],
        postconditions: ["edits the target message content"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.edit_message,
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
                    tool: T.edit_message,
                    summary: "Channel not found or not text-based.",
                    data: null,
                    errorMessage: "Channel not found or not text-based.",
                };
            }
            let message;
            try {
                message = await channel.messages.fetch(String(args.message_id));
            } catch {
                return {
                    tool: T.edit_message,
                    summary: "Message not found.",
                    data: null,
                    errorMessage: "Message not found.",
                };
            }
            if (!message.editable) {
                return {
                    tool: T.edit_message,
                    summary: "Cannot edit this message (not authored by the bot).",
                    data: null,
                    errorMessage:
                        "Cannot edit this message — the bot can only edit its own messages.",
                };
            }
            const edited = await message.edit(String(args.content));
            const channelMention = `<#${channel.id}>`;
            return {
                tool: T.edit_message,
                summary: `Edited message ${edited.id} in ${channelMention}.`,
                data: {
                    messageId: edited.id,
                    channelId: channel.id,
                    channelMention,
                    messageUrl: edited.url,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "✏️", labelPt: "Editar mensagem" },

    describeApproval(args: ToolArguments) {
        return `Editar mensagem ${args.message_id ?? "?"} em <#${args.channel_id ?? "desconhecido"}>`;
    },
};

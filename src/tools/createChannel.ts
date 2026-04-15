import { z } from "zod";
import { ChannelType } from "discord.js";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        name: {
            type: "string",
            description: "The name for the new channel.",
        },
        type: {
            type: "string",
            enum: ["text", "voice"],
            description: "Channel type. Default: text.",
        },
        category_id: {
            type: "string",
            description: "Optional: parent category ID to place the channel under.",
        },
        topic: {
            type: "string",
            description: "Optional: channel topic/description.",
        },
    },
    required: ["name"],
} as const;

export const createChannelTool: ToolDefinition = {
    name: T.create_channel,

    catalog: {
        effect: "write",
        description: "Create a new text or voice channel in the guild.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Create a new channel in the guild. This is a WRITE action that requires admin approval before execution. Use only when explicitly asked to create a channel.",
        parameters,
    },

    capability: {
        description:
            "Create a new channel in the guild. Write action — requires admin approval.",
        inputSchema: z.object({
            name: z.string().describe("Name for the new channel."),
            type: z
                .enum(["text", "voice"])
                .optional()
                .describe("Channel type. Default: text."),
            category_id: z.string().optional().describe("Parent category ID."),
            topic: z.string().optional().describe("Channel topic/description."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["admin"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["creates a new channel in the guild"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.create_channel,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const channelType =
                args.type === "voice"
                    ? ChannelType.GuildVoice
                    : ChannelType.GuildText;
            const options: import("discord.js").GuildChannelCreateOptions = {
                name: String(args.name),
                type: channelType,
            };
            if (
                typeof args.category_id === "string" &&
                args.category_id.trim()
            ) {
                options.parent = args.category_id.trim();
            }
            if (
                typeof args.topic === "string" &&
                args.topic.trim() &&
                channelType === ChannelType.GuildText
            ) {
                options.topic = args.topic.trim();
            }
            const created = await context.guild.channels.create(options);
            const channelMention = `<#${created.id}>`;
            return {
                tool: T.create_channel,
                summary: `Created ${args.type === "voice" ? "voice" : "text"} channel ${channelMention}.`,
                data: {
                    channelId: created.id,
                    channelMention,
                    channelName: created.name,
                    type: args.type || "text",
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "📝", labelPt: "Criar canal" },

    describeApproval(args: ToolArguments) {
        return `Criar canal "${args.name ?? "?"}" no servidor`;
    },
};

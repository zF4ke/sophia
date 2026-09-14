import { z } from "zod";
import { ChannelType } from "discord.js";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        channel_id: {
            type: "string",
            description:
                "The channel ID to edit (from resolve_channel_targets or list_guild_structure).",
        },
        name: {
            type: "string",
            description: "Optional: new name for the channel.",
        },
        topic: {
            type: "string",
            description: "Optional: new topic/description for the channel (text channels only).",
        },
    },
    required: ["channel_id"],
} as const;

export const editChannelTool: ToolDefinition = {
    name: T.edit_channel,

    catalog: {
        effect: "destructive",
        description:
            "Edit a channel's name or topic. Destructive — follows the requester action tier and approval mode.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Edit a channel's name or topic. This is a DESTRUCTIVE action that follows the requester action tier and approval mode. Use only when explicitly asked to rename a channel or change its topic. At least one of `name` or `topic` must be provided.",
        parameters,
    },

    capability: {
        description:
            "Edit a channel's name or topic. Destructive — follows the requester action tier and approval mode.",
        inputSchema: z.object({
            channel_id: z.string().describe("Channel ID to edit."),
            name: z.string().optional().describe("New channel name."),
            topic: z.string().optional().describe("New channel topic (text channels only)."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "destructive",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist", "channel must exist"],
        postconditions: ["edits the channel name and/or topic"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.edit_channel,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const channel = context.guild.channels.cache.get(
                String(args.channel_id),
            );
            if (!channel) {
                return {
                    tool: T.edit_channel,
                    summary: "Channel not found.",
                    data: null,
                    errorMessage: "Channel not found.",
                };
            }
            const hasName = typeof args.name === "string" && args.name.trim();
            const hasTopic = typeof args.topic === "string";
            if (!hasName && !hasTopic) {
                return {
                    tool: T.edit_channel,
                    summary: "Nothing to change — provide at least `name` or `topic`.",
                    data: null,
                    errorMessage: "Provide at least `name` or `topic`.",
                };
            }
            const editOptions: import("discord.js").GuildChannelEditOptions = {};
            if (hasName) {
                editOptions.name = (args.name as string).trim();
            }
            if (
                hasTopic &&
                channel.type === ChannelType.GuildText
            ) {
                editOptions.topic = (args.topic as string).trim() || null;
            }
            const edited = await channel.edit(editOptions);
            const channelMention = `<#${edited.id}>`;
            const changes: string[] = [];
            if (hasName) changes.push(`name → "${edited.name}"`);
            if (hasTopic && channel.type === ChannelType.GuildText) changes.push(`topic updated`);
            return {
                tool: T.edit_channel,
                summary: `Edited ${channelMention}: ${changes.join(", ")}.`,
                data: {
                    channelId: edited.id,
                    channelMention,
                    channelName: edited.name,
                    changes,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "📝", labelPt: "Editar canal" },

    describeApproval(args: ToolArguments) {
        const parts: string[] = [];
        if (args.name) parts.push(`nome → "${args.name}"`);
        if (args.topic !== undefined) parts.push("tópico");
        return `Editar canal <#${args.channel_id ?? "desconhecido"}>: ${parts.join(", ") || "?"}`;
    },
};

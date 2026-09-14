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
                "The channel ID to delete (from resolve_channel_targets or list_guild_structure).",
        },
        reason: {
            type: "string",
            description: "Optional reason for the deletion.",
        },
    },
    required: ["channel_id"],
} as const;

export const deleteChannelTool: ToolDefinition = {
    name: T.delete_channel,

    catalog: {
        effect: "destructive",
        description:
            "Delete a channel from the guild. Destructive — follows the requester action tier and approval mode.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Delete a channel from the guild. This is a DESTRUCTIVE action that follows the requester action tier and approval mode. Use only when explicitly asked to delete a channel.",
        parameters,
    },

    capability: {
        description:
            "Delete a channel from the guild. Destructive — follows the requester action tier and approval mode.",
        inputSchema: z.object({
            channel_id: z.string().describe("Channel ID to delete."),
            reason: z.string().optional().describe("Reason for deletion."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "destructive",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist", "channel must exist"],
        postconditions: ["deletes the channel from the guild"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.delete_channel,
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
                    tool: T.delete_channel,
                    summary: "Channel not found.",
                    data: null,
                    errorMessage: "Channel not found.",
                };
            }
            const channelName = channel.name;
            const channelId = channel.id;
            const reason =
                typeof args.reason === "string" ? args.reason : undefined;
            await channel.delete(reason);
            return {
                tool: T.delete_channel,
                summary: `Deleted channel #${channelName} (${channelId}).`,
                data: { channelId, channelName, deleted: true },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "🗑️", labelPt: "Eliminar canal" },

    describeApproval(args: ToolArguments) {
        return `Eliminar canal <#${args.channel_id ?? "desconhecido"}>`;
    },
};

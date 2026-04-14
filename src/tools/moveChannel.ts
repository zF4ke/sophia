import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        channel_id: {
            type: "string",
            description: "The channel ID to move.",
        },
        category_id: {
            type: "string",
            description:
                "Target category ID to move the channel into. Use null or omit to remove from category.",
        },
        position: {
            type: "number",
            description:
                "New position index within the category or guild. 0 = top.",
        },
        lock_permissions: {
            type: "boolean",
            description:
                "Whether to sync permissions with the new category. Default: true.",
        },
    },
    required: ["channel_id"],
} as const;

export const moveChannelTool: ToolDefinition = {
    name: T.move_channel,

    catalog: {
        effect: "write",
        description:
            "Move a channel to a new position or category. Write — requires admin approval.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Move a channel to a different position or category. This is a WRITE action that requires admin approval. Can be used to reorder channels or move them into/out of categories.",
        parameters,
    },

    capability: {
        description:
            "Move a channel to a different position or category. Write — requires admin approval.",
        inputSchema: z.object({
            channel_id: z.string().describe("Channel ID to move."),
            category_id: z
                .string()
                .optional()
                .describe("Target category ID."),
            position: z.number().optional().describe("New position index."),
            lock_permissions: z
                .boolean()
                .optional()
                .describe("Sync permissions with category. Default: true."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["admin"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: [
            "moves the channel to the specified position/category",
        ],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.move_channel,
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
                    tool: T.move_channel,
                    summary: "Channel not found.",
                    data: null,
                    errorMessage: "Channel not found.",
                };
            }
            const updates: Record<string, unknown> = {};
            const lockPerms = args.lock_permissions !== false;
            if (
                typeof args.category_id === "string" &&
                args.category_id.trim()
            ) {
                updates.parent = args.category_id.trim();
                updates.lockPermissions = lockPerms;
            }
            if (typeof args.position === "number") {
                updates.position = args.position;
            }
            await channel.edit(updates);
            const channelMention = `<#${channel.id}>`;
            const categoryName = channel.parent?.name ?? "none";
            return {
                tool: T.move_channel,
                summary: `Moved ${channelMention} to category "${categoryName}"${typeof args.position === "number" ? ` at position ${args.position}` : ""}.`,
                data: {
                    channelId: channel.id,
                    channelMention,
                    channelName: channel.name,
                    newCategoryId: channel.parentId ?? null,
                    newCategoryName: categoryName,
                    newPosition:
                        "position" in channel ? channel.position : null,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "📦", labelPt: "Mover canal" },

    describeApproval(args: ToolArguments) {
        return `Mover canal <#${args.channel_id ?? "desconhecido"}>${args.category_id ? ` para categoria` : ""}${typeof args.position === "number" ? ` para posição ${args.position}` : ""}`;
    },
};

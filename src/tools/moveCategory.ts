import { ChannelType } from "discord.js";
import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        category_id: {
            type: "string",
            description: "The category ID to move.",
        },
        position: {
            type: "number",
            description:
                "New position index among categories in the guild. 0 = top.",
        },
    },
    required: ["category_id", "position"],
} as const;

export const moveCategoryTool: ToolDefinition = {
    name: T.move_category,

    catalog: {
        effect: "write",
        description:
            "Move a category to a new position in the guild. Write — follows the requester action tier and approval mode.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Move a category to a different position in the guild channel list. This is a WRITE action that follows the requester action tier and approval mode. Use to reorder categories.",
        parameters,
    },

    capability: {
        description:
            "Move a category to a different position in the guild. Write — follows the requester action tier and approval mode.",
        inputSchema: z.object({
            category_id: z.string().describe("Category ID to move."),
            position: z.number().describe("New position index. 0 = top."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: [
            "moves the category to the specified position in the guild",
        ],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.move_category,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const channel = context.guild.channels.cache.get(
                String(args.category_id),
            );
            if (!channel || channel.type !== ChannelType.GuildCategory) {
                return {
                    tool: T.move_category,
                    summary: "Category not found.",
                    data: null,
                    errorMessage: "Category not found or the ID does not point to a category.",
                };
            }
            await channel.edit({ position: Number(args.position) });
            const channelMention = `<#${channel.id}>`;
            return {
                tool: T.move_category,
                summary: `Moved category "${channel.name}" (${channelMention}) to position ${args.position}.`,
                data: {
                    categoryId: channel.id,
                    categoryMention: channelMention,
                    categoryName: channel.name,
                    newPosition: channel.position,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "📂", labelPt: "Mover categoria" },

    describeApproval(args: ToolArguments) {
        return `Mover categoria <#${args.category_id ?? "desconhecido"}> para posição ${args.position ?? "?"}`;
    },
};

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
            description: "The name for the new category.",
        },
        position: {
            type: "number",
            description: "Optional category position. 0 = top.",
        },
    },
    required: ["name"],
} as const;

export const createCategoryTool: ToolDefinition = {
    name: T.create_category,

    catalog: {
        effect: "write",
        description: "Create a new category in the guild.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Create a new category in the guild. This is a WRITE action that follows the requester action tier and approval mode before execution. Use only when explicitly asked to create a category.",
        parameters,
    },

    capability: {
        description:
            "Create a new category in the guild. Write action - follows the requester action tier and approval mode.",
        inputSchema: z.object({
            name: z.string().describe("Name for the new category."),
            position: z
                .number()
                .optional()
                .describe("Optional category position. 0 = top."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["creates a new category in the guild"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.create_category,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }

            const options: import("discord.js").GuildChannelCreateOptions = {
                name: String(args.name),
                type: ChannelType.GuildCategory,
            };

            if (typeof args.position === "number") {
                options.position = args.position;
            }

            const created = await context.guild.channels.create(options);
            const categoryMention = `<#${created.id}>`;
            return {
                tool: T.create_category,
                summary: `Created category ${categoryMention}.`,
                data: {
                    categoryId: created.id,
                    categoryMention,
                    categoryName: created.name,
                    position:
                        typeof args.position === "number"
                            ? args.position
                            : null,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "📁", labelPt: "Criar categoria" },

    describeApproval(args: ToolArguments) {
        return `Criar categoria "${args.name ?? "?"}" no servidor`;
    },
};
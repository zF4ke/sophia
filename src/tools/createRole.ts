import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        name: {
            type: "string",
            description: "The name for the new role.",
        },
        color: {
            type: "string",
            description:
                "Optional: hex color for the role (e.g. '#FF0000'). Default: no color.",
        },
        hoist: {
            type: "string",
            enum: ["true", "false"],
            description:
                "Optional: whether to display the role separately in the sidebar. Default: false.",
        },
        mentionable: {
            type: "string",
            enum: ["true", "false"],
            description:
                "Optional: whether anyone can @mention this role. Default: false.",
        },
        reason: {
            type: "string",
            description: "Optional: reason for creating the role (audit log).",
        },
    },
    required: ["name"],
} as const;

export const createRoleTool: ToolDefinition = {
    name: T.create_role,

    catalog: {
        effect: "write",
        description:
            "Create a new role in the guild. Write — requires admin approval.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Create a new role in the guild. This is a WRITE action that requires admin approval. Use only when explicitly asked to create a role.",
        parameters,
    },

    capability: {
        description:
            "Create a new role in the guild. Write — requires admin approval.",
        inputSchema: z.object({
            name: z.string().describe("Name for the new role."),
            color: z
                .string()
                .optional()
                .describe("Hex color (e.g. '#FF0000')."),
            hoist: z
                .string()
                .optional()
                .describe("'true' to display separately in sidebar."),
            mentionable: z
                .string()
                .optional()
                .describe("'true' to allow anyone to @mention."),
            reason: z
                .string()
                .optional()
                .describe("Audit log reason."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["admin"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["creates a new role in the guild"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.create_role,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const options: import("discord.js").RoleCreateOptions = {
                name: String(args.name),
            };
            if (typeof args.color === "string" && args.color.trim()) {
                options.color = args.color.trim() as import("discord.js").ColorResolvable;
            }
            if (args.hoist === "true") {
                options.hoist = true;
            }
            if (args.mentionable === "true") {
                options.mentionable = true;
            }
            if (typeof args.reason === "string" && args.reason.trim()) {
                options.reason = args.reason.trim();
            }
            const created = await context.guild.roles.create(options);
            const roleMention = `<@&${created.id}>`;
            return {
                tool: T.create_role,
                summary: `Created role ${roleMention} ("${created.name}").`,
                data: {
                    roleId: created.id,
                    roleMention,
                    roleName: created.name,
                    color: created.hexColor,
                    hoist: created.hoist,
                    mentionable: created.mentionable,
                    position: created.position,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "🎭", labelPt: "Criar cargo" },

    describeApproval(args: ToolArguments) {
        return `Criar cargo "${args.name ?? "?"}" no servidor`;
    },
};

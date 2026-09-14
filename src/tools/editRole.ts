import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        role_id: {
            type: "string",
            description:
                "The role ID to edit (from get_role_info or list_roles).",
        },
        name: {
            type: "string",
            description: "Optional: new name for the role.",
        },
        color: {
            type: "string",
            description:
                "Optional: new hex color (e.g. '#00FF00'). Use '#000000' to remove color.",
        },
        hoist: {
            type: "string",
            enum: ["true", "false"],
            description: "Optional: whether to display the role separately in the sidebar.",
        },
        mentionable: {
            type: "string",
            enum: ["true", "false"],
            description: "Optional: whether anyone can @mention this role.",
        },
        reason: {
            type: "string",
            description: "Optional: reason for editing the role (audit log).",
        },
    },
    required: ["role_id"],
} as const;

export const editRoleTool: ToolDefinition = {
    name: T.edit_role,

    catalog: {
        effect: "destructive",
        description:
            "Edit an existing role in the guild. Destructive — follows the requester action tier and approval mode.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Edit an existing role's name, color, hoist, or mentionable status. This is a DESTRUCTIVE action that follows the requester action tier and approval mode. Use only when explicitly asked to edit a role. At least one of `name`, `color`, `hoist`, or `mentionable` must be provided.",
        parameters,
    },

    capability: {
        description:
            "Edit an existing role in the guild. Destructive — follows the requester action tier and approval mode.",
        inputSchema: z.object({
            role_id: z.string().describe("Role ID to edit."),
            name: z.string().optional().describe("New role name."),
            color: z.string().optional().describe("New hex color."),
            hoist: z.string().optional().describe("'true' or 'false'."),
            mentionable: z.string().optional().describe("'true' or 'false'."),
            reason: z.string().optional().describe("Audit log reason."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "destructive",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist", "role must exist"],
        postconditions: ["edits the role properties"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.edit_role,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const role = context.guild.roles.cache.get(String(args.role_id));
            if (!role) {
                return {
                    tool: T.edit_role,
                    summary: "Role not found.",
                    data: null,
                    errorMessage: "Role not found.",
                };
            }
            if (role.managed) {
                return {
                    tool: T.edit_role,
                    summary: "Cannot edit a managed/integration role.",
                    data: null,
                    errorMessage:
                        "This role is managed by an integration and cannot be edited.",
                };
            }

            const hasName = typeof args.name === "string" && args.name.trim();
            const hasColor = typeof args.color === "string" && args.color.trim();
            const hasHoist = args.hoist === "true" || args.hoist === "false";
            const hasMentionable =
                args.mentionable === "true" || args.mentionable === "false";

            if (!hasName && !hasColor && !hasHoist && !hasMentionable) {
                return {
                    tool: T.edit_role,
                    summary:
                        "Nothing to change — provide at least `name`, `color`, `hoist`, or `mentionable`.",
                    data: null,
                    errorMessage:
                        "Provide at least one property to change.",
                };
            }

            const editOptions: import("discord.js").RoleEditOptions = {};
            if (hasName) editOptions.name = (args.name as string).trim();
            if (hasColor)
                editOptions.color = (args.color as string).trim() as import("discord.js").ColorResolvable;
            if (hasHoist) editOptions.hoist = args.hoist === "true";
            if (hasMentionable)
                editOptions.mentionable = args.mentionable === "true";
            if (typeof args.reason === "string" && args.reason.trim()) {
                editOptions.reason = args.reason.trim();
            }

            const edited = await role.edit(editOptions);
            const roleMention = `<@&${edited.id}>`;
            const changes: string[] = [];
            if (hasName) changes.push(`name → "${edited.name}"`);
            if (hasColor) changes.push(`color → ${edited.hexColor}`);
            if (hasHoist)
                changes.push(`hoist → ${edited.hoist}`);
            if (hasMentionable)
                changes.push(`mentionable → ${edited.mentionable}`);

            return {
                tool: T.edit_role,
                summary: `Edited role ${roleMention}: ${changes.join(", ")}.`,
                data: {
                    roleId: edited.id,
                    roleMention,
                    roleName: edited.name,
                    color: edited.hexColor,
                    hoist: edited.hoist,
                    mentionable: edited.mentionable,
                    position: edited.position,
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

    display: { icon: "🎨", labelPt: "Editar cargo" },

    describeApproval(args: ToolArguments) {
        const parts: string[] = [];
        if (args.name) parts.push(`nome → "${args.name}"`);
        if (args.color) parts.push(`cor → ${args.color}`);
        if (args.hoist !== undefined) parts.push("destaque");
        if (args.mentionable !== undefined) parts.push("mencionável");
        return `Editar cargo <@&${args.role_id ?? "desconhecido"}>: ${parts.join(", ") || "?"}`;
    },
};

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
                "The role ID to delete (from get_role_info or list_roles).",
        },
        reason: {
            type: "string",
            description: "Optional reason for the deletion (audit log).",
        },
    },
    required: ["role_id"],
} as const;

export const deleteRoleTool: ToolDefinition = {
    name: T.delete_role,

    catalog: {
        effect: "destructive",
        description:
            "Delete a role from the guild. Destructive — requires admin approval.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Delete a role from the guild. This is a DESTRUCTIVE action that requires admin approval. Use only when explicitly asked to delete a role.",
        parameters,
    },

    capability: {
        description:
            "Delete a role from the guild. Destructive — requires admin approval.",
        inputSchema: z.object({
            role_id: z.string().describe("Role ID to delete."),
            reason: z.string().optional().describe("Audit log reason."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "destructive",
        authRequirements: ["admin"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist", "role must exist"],
        postconditions: ["deletes the role from the guild"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.delete_role,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const role = context.guild.roles.cache.get(
                String(args.role_id),
            );
            if (!role) {
                return {
                    tool: T.delete_role,
                    summary: "Role not found.",
                    data: null,
                    errorMessage: "Role not found.",
                };
            }
            if (role.managed) {
                return {
                    tool: T.delete_role,
                    summary: "Cannot delete a managed/integration role.",
                    data: null,
                    errorMessage:
                        "This role is managed by an integration and cannot be deleted.",
                };
            }
            const roleName = role.name;
            const roleId = role.id;
            const reason =
                typeof args.reason === "string" ? args.reason : undefined;
            await role.delete(reason);
            return {
                tool: T.delete_role,
                summary: `Deleted role "${roleName}" (${roleId}).`,
                data: { roleId, roleName, deleted: true },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "🗑️", labelPt: "Eliminar cargo" },

    describeApproval(args: ToolArguments) {
        return `Eliminar cargo <@&${args.role_id ?? "desconhecido"}>`;
    },
};

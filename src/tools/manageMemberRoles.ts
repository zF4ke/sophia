import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        member_id: {
            type: "string",
            description:
                "The member's Discord ID (from resolve_member_identity).",
        },
        add_roles: {
            type: "array",
            items: { type: "string" },
            description: "Array of role IDs to add to the member.",
        },
        remove_roles: {
            type: "array",
            items: { type: "string" },
            description: "Array of role IDs to remove from the member.",
        },
        reason: {
            type: "string",
            description: "Optional reason for the role change.",
        },
    },
    required: ["member_id"],
} as const;

export const manageMemberRolesTool: ToolDefinition = {
    name: T.manage_member_roles,

    catalog: {
        effect: "destructive",
        description:
            "Add or remove roles from a guild member. Access-sensitive change; follows the requester action tier and approval mode.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Add or remove roles from a guild member. This is a access-sensitive action requiring the destructive tier that follows the requester action tier and approval mode. Use only when explicitly asked to modify someone's roles.",
        parameters,
    },

    capability: {
        description:
            "Add or remove roles from a guild member. Access-sensitive change; follows the requester action tier and approval mode.",
        inputSchema: z.object({
            member_id: z.string().describe("Member's Discord ID."),
            add_roles: z
                .array(z.string())
                .optional()
                .describe("Role IDs to add."),
            remove_roles: z
                .array(z.string())
                .optional()
                .describe("Role IDs to remove."),
            reason: z.string().optional().describe("Reason for change."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "destructive",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist", "member must exist"],
        postconditions: ["adds/removes roles from the member"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.manage_member_roles,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const member = await context.guild.members
                .fetch({ user: String(args.member_id), force: true })
                .catch(() => null);
            if (!member) {
                return {
                    tool: T.manage_member_roles,
                    summary: "Member not found.",
                    data: null,
                    errorMessage: "Member not found.",
                };
            }
            const reason =
                typeof args.reason === "string" ? args.reason : undefined;
            const added: string[] = [];
            const removed: string[] = [];
            const addedRoleIds: string[] = [];
            const removedRoleIds: string[] = [];
            const addRoles = Array.isArray(args.add_roles)
                ? args.add_roles.filter(
                      (r): r is string => typeof r === "string",
                  )
                : [];
            const removeRoles = Array.isArray(args.remove_roles)
                ? args.remove_roles.filter(
                      (r): r is string => typeof r === "string",
                  )
                : [];

            if (!addRoles.length && !removeRoles.length) throw new Error("Provide at least one role to add or remove.");
            if (addRoles.some(id => removeRoles.includes(id))) throw new Error("A role cannot be added and removed in the same action.");
            const roles = new Map<string, import("discord.js").Role>();
            for (const roleId of new Set([...addRoles, ...removeRoles])) {
                const role = await context.guild.roles.fetch(roleId, { force: true });
                if (!role || role.managed || !role.editable || role.id === context.guild.id) throw new Error(`Role ${roleId} is unavailable or cannot be managed by Sophia.`);
                roles.set(roleId, role);
            }
            for (const roleId of new Set(addRoles)) {
                const role = roles.get(roleId);
                if (role && !member.roles.cache.has(roleId)) {
                    await member.roles.add(role, reason);
                    added.push(role.name);
                    addedRoleIds.push(role.id);
                }
            }
            for (const roleId of new Set(removeRoles)) {
                const role = roles.get(roleId);
                if (role && member.roles.cache.has(roleId)) {
                    await member.roles.remove(role, reason);
                    removed.push(role.name);
                    removedRoleIds.push(role.id);
                }
            }
            const parts: string[] = [];
            if (added.length) parts.push(`added ${added.join(", ")}`);
            if (removed.length) parts.push(`removed ${removed.join(", ")}`);
            const summary = parts.length
                ? `<@${member.id}>: ${parts.join("; ")}. Added IDs: ${addedRoleIds.join(", ") || "none"}; removed IDs: ${removedRoleIds.join(", ") || "none"}.`
                : `No role changes for <@${member.id}>.`;
            return {
                tool: T.manage_member_roles,
                summary,
                data: {
                    memberId: member.id,
                    memberMention: `<@${member.id}>`,
                    addedRoleIds,
                    removedRoleIds,
                    addedRoleMentions: addedRoleIds.map(id => `<@&${id}>`),
                    removedRoleMentions: removedRoleIds.map(id => `<@&${id}>`),
                    displayName: member.displayName,
                    rolesAdded: added,
                    rolesRemoved: removed,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "🏷️", labelPt: "Gerir cargos" },

    describeApproval(args: ToolArguments) {
        return `Alterar cargos de <@${args.member_id ?? "desconhecido"}>`;
    },
};

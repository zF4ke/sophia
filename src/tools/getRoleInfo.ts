import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { DiscordToolResult } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        role: {
            type: "string",
            description: "Role name, ID, or mention to look up.",
        },
    },
    required: ["role"],
} as const;

export const getRoleInfoTool: ToolDefinition = {
    name: T.get_role_info,

    catalog: {
        effect: "read",
        description: "Fetch detailed information about a Discord role by name or ID.",
        evidenceRole: "live_evidence",
    },

    schema: {
        description:
            "Fetch detailed information about a Discord role by name or ID. Returns name, color, position, permissions, member count, mentionable status, hoisted status, managed status, icon, and which members have the role.",
        parameters,
    },

    capability: {
        description: "Fetch detailed information about a Discord role by name or ID.",
        inputSchema: z.object({
            role: z.string().describe("Role name, ID, or mention."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["returns detailed role information"],
        async run(context, args) {
            if (!context.guild) {
                return { tool: T.get_role_info, summary: "No guild context.", data: null, errorMessage: "No guild context." };
            }
            const query = String(args.role || "").trim();
            const mentionMatch = query.match(/^<@&(\d+)>$/);
            const roleId = mentionMatch ? mentionMatch[1] : (/^\d{6,25}$/.test(query) ? query : null);

            await context.guild.roles.fetch();
            let role = roleId ? context.guild.roles.cache.get(roleId) ?? null : null;
            if (!role) {
                const lower = query.toLowerCase();
                role = context.guild.roles.cache.find(
                    (r) => r.name.toLowerCase() === lower || r.name.toLowerCase().includes(lower)
                ) ?? null;
            }
            if (!role) {
                return { tool: T.get_role_info, summary: `Role "${query}" not found.`, data: null, errorMessage: `Role "${query}" not found.` };
            }

            const members = role.members.map((m) => ({
                id: m.id,
                displayName: m.displayName,
                username: m.user.username,
            }));

            const permissionFlags = role.permissions.toArray();
            return {
                tool: T.get_role_info,
                summary: `${role.name}: ${members.length} members, position ${role.position}, color ${role.hexColor}.`,
                data: {
                    id: role.id,
                    name: role.name,
                    color: role.hexColor,
                    position: role.position,
                    hoist: role.hoist,
                    managed: role.managed,
                    mentionable: role.mentionable,
                    permissions: permissionFlags,
                    icon: role.iconURL() ?? null,
                    unicodeEmoji: role.unicodeEmoji ?? null,
                    createdAt: role.createdAt.toISOString(),
                    memberCount: members.length,
                    members: members.slice(0, 50),
                    roleMention: `<@&${role.id}>`,
                },
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data) return [];
            const item = run.data as Record<string, unknown>;
            return [
                {
                    tool: T.get_role_info,
                    summary: run.summary,
                    content: `Role ${String(item.name || "?")}: ${String(item.memberCount || 0)} members, position ${String(item.position || 0)}, color ${String(item.color || "#000000")}`,
                    evidenceRole: "live_evidence" as const,
                    strength: "metadata" as const,
                    sourceOrigin: "none" as const,
                },
            ];
        },
    },

    display: { icon: "🎭", labelPt: "Ler info de cargo" },
};

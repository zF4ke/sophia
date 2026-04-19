import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { DiscordToolResult } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        include_member_counts: {
            type: "string",
            enum: ["true", "false"],
            description:
                "Optional: include number of members per role. Default: true.",
        },
    },
    required: [],
} as const;

export const listRolesTool: ToolDefinition = {
    name: T.list_roles,

    catalog: {
        effect: "read",
        description: "List all roles in the guild with their IDs, colors, and positions.",
        evidenceRole: "live_evidence",
    },

    schema: {
        description:
            "List all roles in the guild sorted by position (highest first). Returns each role's ID, name, color, member count, hoist status, mentionable status, and whether it is managed. Use when the user asks about available roles or needs to pick a role by name.",
        parameters,
    },

    capability: {
        description:
            "List all roles in the guild sorted by position. Use when the user asks about available roles.",
        inputSchema: z.object({
            include_member_counts: z
                .string()
                .optional()
                .describe("'true' (default) or 'false'."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["returns list of all guild roles"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.list_roles,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            await context.guild.roles.fetch();
            const includeCounts = args.include_member_counts !== "false";

            const roles = context.guild.roles.cache
                .filter((r) => r.id !== context.guild!.id) // exclude @everyone
                .sort((a, b) => b.position - a.position)
                .map((r) => {
                    const entry: Record<string, unknown> = {
                        id: r.id,
                        name: r.name,
                        color: r.hexColor,
                        position: r.position,
                        hoist: r.hoist,
                        mentionable: r.mentionable,
                        managed: r.managed,
                        roleMention: `<@&${r.id}>`,
                    };
                    if (includeCounts) {
                        entry.memberCount = r.members.size;
                    }
                    return entry;
                });

            return {
                tool: T.list_roles,
                summary: `${roles.length} role(s) in the guild.`,
                data: {
                    roles,
                    totalCount: roles.length,
                },
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data || typeof run.data !== "object") return [];
            const payload = run.data as {
                roles?: Array<Record<string, unknown>>;
            };
            const roles = payload.roles || [];

            return roles.slice(0, 15).map((item) => ({
                tool: T.list_roles,
                summary: run.summary,
                content: `${String(item.name || "?")} (id=${String(item.id || "?")}, color=${String(item.color || "#000000")}, members=${String(item.memberCount ?? "?")}, pos=${String(item.position ?? "?")})`,
                evidenceRole: "live_evidence" as const,
                strength: "metadata" as const,
                sourceOrigin: "none" as const,
            }));
        },
    },

    display: { icon: "🏷️", labelPt: "Listar cargos" },
};

import { z } from "zod";
import { T } from "@/shared/discordTools";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import type { DiscordToolResult } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        filters: {
            type: "string",
            description: "Optional name or username fragment to filter by.",
        },
        limit: {
            type: "number",
            description: "Page size. Default 20.",
        },
        offset: {
            type: "number",
            description: "Pagination offset.",
        },
    },
    required: [],
} as const;

export const listMembersTool: ToolDefinition = {
    name: T.list_members,

    catalog: {
        effect: "read",
        description: "List guild members live from Discord, optionally filtered.",
        evidenceRole: "live_evidence",
    },

    schema: {
        description:
            "List live guild members sorted by join date with their IDs, usernames, display names, and roles. Supports pagination with offset and optional name/username fragment filter.",
        parameters,
    },

    capability: {
        description:
            "List live guild members sorted by join date. Pass filters to narrow by name/username fragment, or omit filters to list all members in pages.",
        inputSchema: z.object({
            filters: z.string().describe("Optional name or username fragment. Omit to list all members.").optional(),
            limit: z.number().int().positive().describe("Page size, default 20.").optional(),
            offset: z.number().int().nonnegative().describe("Pagination offset.").optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["returns live member list data"],
        async run(context, args) {
            const members = await DiscordLiveService.listMembers(context.guild, {
                filters: typeof args.filters === "string" ? args.filters : undefined,
                limit: typeof args.limit === "number" ? args.limit : undefined,
                offset: typeof args.offset === "number" ? args.offset : undefined,
                sort: "joined_at",
            });
            return {
                tool: T.list_members,
                summary: members.returnedCount
                    ? members.hasMore
                        ? `Showing ${members.returnedCount} of ${members.totalCount} members in join order.`
                        : `${members.returnedCount} members listed in join order.`
                    : "No matching members found.",
                data: members,
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data || typeof run.data !== "object") return [];
            const payload = run.data as {
                members?: Array<Record<string, unknown>>;
                hasMore?: boolean;
                totalCount?: number;
                offset?: number;
            };
            const members = payload.members || [];

            const evidence = members.slice(0, 10).map((item) => {
                const parts = [
                    `${String(item.displayName || "No Display Name")} (@${String(item.username || "no username")})`,
                ];
                if (item.id != null) parts.push(`id=${String(item.id)}`);
                if (item.nickname) parts.push(`nick=${String(item.nickname)}`);
                if (item.joinedTimestamp) {
                    parts.push(`joined=${new Date(Number(item.joinedTimestamp)).toISOString()}`);
                }
                if (item.isBot) parts.push("bot=true");

                return {
                    tool: T.list_members,
                    summary: run.summary,
                    content: parts.join("; "),
                    evidenceRole: "live_evidence" as const,
                    strength: "metadata" as const,
                    sourceOrigin: "none" as const,
                    authorId: item.id == null ? null : String(item.id),
                    authorName: item.displayName == null ? null : String(item.displayName),
                };
            });

            if (payload.hasMore) {
                evidence.push({
                    tool: T.list_members,
                    summary: run.summary,
                    content: `...and ${(payload.totalCount || 0) - members.length} more members (use offset=${(payload.offset || 0) + members.length} to continue).`,
                    evidenceRole: "live_evidence" as const,
                    strength: "metadata" as const,
                    sourceOrigin: "none" as const,
                    authorId: null,
                    authorName: null,
                });
            }

            return evidence;
        },
    },

    display: { icon: "👥", labelPt: "Listar membros" },
};

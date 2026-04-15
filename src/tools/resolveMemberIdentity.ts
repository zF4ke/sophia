import { z } from "zod";
import { T } from "@/shared/discordTools";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import type { DiscordToolResult, ResolvedMemberIdentity } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        queries: {
            type: "array",
            items: { type: "string" },
            description: "One or more member names, usernames, nicknames, or Discord snowflake IDs to resolve. Pass all queries in a single call instead of calling this tool multiple times.",
        },
    },
    required: ["queries"],
} as const;

export const resolveMemberIdentityTool: ToolDefinition = {
    name: T.resolve_member_identity,

    catalog: {
        effect: "read",
        description:
            "Resolve a member or bot in the current guild using exact ids, live guild fetches, and same-guild historical author fallback.",
        evidenceRole: "live_evidence",
    },

    schema: {
        description:
            "Resolve one or more members or bots in the current guild. Pass an array of Discord snowflake IDs, usernames, display names, or nicknames. Returns each member's resolved identity with their Discord ID, username, display name, roles, and guild membership status. Always pass all queries in a single call — do not call this tool once per member.",
        parameters,
    },

    capability: {
        description:
            "Resolve members or bots in the current guild using exact ids, live guild fetches, and same-guild historical message authors. Pass Discord snowflake IDs, usernames, display names, or nicknames — snowflake IDs resolve directly without a search.",
        inputSchema: z.object({
            queries: z.array(z.string()).describe("Member names, usernames, nicknames, or Discord snowflake IDs."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["returns the best resolved member identities for the current guild"],
        async run(context, args) {
            const rawQueries = Array.isArray(args.queries) ? args.queries.map(String) : [String(args.queries || context.question)];
            const rawResults = await Promise.all(
                rawQueries.map((q) => DiscordLiveService.resolveMemberIdentity(context.guild, q))
            );
            const results = rawResults.map((r, i) =>
                r ?? { query: rawQueries[i], resolvedId: null, notFound: true }
            );
            const resolved = rawResults.filter((r): r is ResolvedMemberIdentity => r !== null);
            const summaryParts = rawQueries.map((q, i) => {
                const r = rawResults[i];
                if (!r) return `${q}: not found`;
                return r.isCurrentGuildMember
                    ? `${r.displayName} (@${r.username}): resolved`
                    : `${r.displayName}: historical only`;
            });
            return {
                tool: T.resolve_member_identity,
                summary: `Resolved ${resolved.length}/${rawQueries.length} member(s): ${summaryParts.join("; ")}.`,
                data: { results },
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data) return [];

            const items = extractMemberResults(run.data);
            return items.map((item) => {
                const currentState =
                    item.isCurrentGuildMember === false ? "historical guild memory" : "current guild";
                const resolvedId =
                    item.resolvedId == null
                        ? item.id == null ? null : String(item.id)
                        : String(item.resolvedId);
                const resolveParts = [
                    `${String(item.displayName || "No Display Name")} (@${String(item.username || "no username")})`,
                ];
                if (resolvedId) resolveParts.push(`id=${resolvedId}`);
                if (item.nickname) resolveParts.push(`nick=${String(item.nickname)}`);
                resolveParts.push(`from ${currentState}`);

                return {
                    tool: T.resolve_member_identity,
                    summary: run.summary,
                    content: resolveParts.join("; "),
                    evidenceRole: "live_evidence" as const,
                    strength: "metadata" as const,
                    sourceOrigin: "none" as const,
                    authorId: resolvedId,
                    authorName: item.displayName == null ? null : String(item.displayName),
                };
            });
        },

        extractResolvedMember(run: DiscordToolResult): ResolvedMemberIdentity | null {
            if (!run.data || typeof run.data !== "object") return null;

            const items = extractMemberResults(run.data);
            if (!items.length) return null;

            // Return the first resolved member (most common case)
            const item = items[0];
            const resolvedId =
                item.resolvedId == null
                    ? item.id == null ? null : String(item.id)
                    : String(item.resolvedId);
            if (!resolvedId) return null;

            return {
                query: String(item.query || resolvedId),
                resolvedId,
                displayName: String(item.displayName || "No Display Name"),
                username: String(item.username || "no username"),
                globalName: item.globalName == null ? null : String(item.globalName),
                nickname: item.nickname == null ? null : String(item.nickname),
                isBot: Boolean(item.isBot),
                isCurrentGuildMember:
                    item.isCurrentGuildMember == null ? true : Boolean(item.isCurrentGuildMember),
                source:
                    item.source === "historical_author" ||
                    item.source === "live_exact" ||
                    item.source === "live_search" ||
                    item.source === "live_id"
                        ? item.source
                        : "live_id",
                confidence:
                    item.confidence === "high" ||
                    item.confidence === "medium" ||
                    item.confidence === "exact"
                        ? item.confidence
                        : "medium",
                roles: Array.isArray(item.roles) ? item.roles.map(String) : [],
            };
        },
    },

    display: { icon: "🪪", labelPt: "Encontrar membro" },
};

function extractMemberResults(data: unknown): Array<Record<string, unknown>> {
    if (!data || typeof data !== "object") return [];
    const obj = data as Record<string, unknown>;
    // Batched format: { results: [...] }
    if (Array.isArray(obj.results)) {
        return (obj.results as unknown[]).filter((r): r is Record<string, unknown> => r !== null && typeof r === "object");
    }
    // Legacy single-result format
    if (obj.resolvedId !== undefined || obj.query !== undefined) {
        return [obj as Record<string, unknown>];
    }
    return [];
}

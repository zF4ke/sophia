import type { SearchMessageScope } from "@/memory/types";

export class SearchScopeClause {
    public static build(scope: SearchMessageScope): { sql: string; params: string[] } {
        const clauses: string[] = [];
        const params: string[] = [];

        if (scope.guildId) {
            clauses.push(`AND mc.guild_id = ?`);
            params.push(scope.guildId);
        }

        if (scope.channelIds?.length) {
            clauses.push(`AND mc.channel_id IN (${scope.channelIds.map(() => "?").join(", ")})`);
            params.push(...scope.channelIds);
        }

        if (scope.authorIds?.length) {
            clauses.push(`AND m.author_id IN (${scope.authorIds.map(() => "?").join(", ")})`);
            params.push(...scope.authorIds);
        }

        return {
            sql: clauses.join("\n"),
            params,
        };
    }

    public static toFtsQuery(query: string): string {
        return query
            .replace(/[^\p{L}\p{N}_]+/gu, " ")
            .split(/\s+/)
            .map((term) =>
                term
                    .replace(/[^\p{L}\p{N}_]/gu, "")
                    .trim()
            )
            .filter((term) => term.length > 1)
            .filter((term, index, terms) => terms.indexOf(term) === index)
            .map((term) => `"${term.replace(/"/g, '""')}"`)
            .join(" OR ");
    }
}

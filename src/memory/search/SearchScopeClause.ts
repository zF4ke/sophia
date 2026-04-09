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

        return {
            sql: clauses.join("\n"),
            params,
        };
    }

    public static toFtsQuery(query: string): string {
        return query
            .split(/\s+/)
            .map((term) => term.replace(/["']/g, "").trim())
            .filter((term) => term.length > 1)
            .join(" OR ");
    }
}

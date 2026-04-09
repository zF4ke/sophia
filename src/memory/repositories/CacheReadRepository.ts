import { MemoryDatabase } from "@/memory/MemoryDatabase";
import type {
    CachedToolResultRecord,
    ConversationResolutionContextRecord,
    ReusableGroundedContextRecord,
} from "@/memory/types";

export class CacheReadRepository {
    public static getReusableGroundedContext(options: {
        guildId: string | null;
        currentChannelId: string | null;
        questionFingerprint: string;
        routeIntent: string;
        requireSufficient?: boolean;
        currentResponseOrdinal?: number | null;
        maxResponsesAgo?: number;
        now?: number;
    }): ReusableGroundedContextRecord | null {
        const now = options.now ?? Date.now();
        const rows = MemoryDatabase.get()
            .prepare(
                `
                    SELECT *
                    FROM reusable_grounded_contexts
                    WHERE guild_id IS ?
                      AND question_fingerprint = ?
                      AND route_intent = ?
                      AND expiry_timestamp > ?
                      AND (
                        created_response_ordinal IS NULL
                        OR ? IS NULL
                        OR (? - created_response_ordinal) <= ?
                      )
                      ${options.requireSufficient ? "AND sufficient = 1" : ""}
                    ORDER BY
                        CASE
                            WHEN channel_id = ? THEN 0
                            WHEN channel_id IS NULL THEN 2
                            ELSE 1
                        END,
                        created_timestamp DESC
                    LIMIT 1
                `
            )
            .all(
                options.guildId,
                options.questionFingerprint,
                options.routeIntent,
                now,
                options.currentResponseOrdinal ?? null,
                options.currentResponseOrdinal ?? null,
                options.maxResponsesAgo ?? Number.MAX_SAFE_INTEGER,
                options.currentChannelId
            ) as Array<Record<string, unknown>>;

        const row = rows[0];
        return row ? this.mapReusableGroundedContext(row) : null;
    }

    public static getCachedToolResult(
        cacheKey: string,
        currentResponseOrdinal: number | null,
        maxResponsesAgo: number,
        now = Date.now()
    ): CachedToolResultRecord | null {
        const row = MemoryDatabase.get()
            .prepare(
                `
                    SELECT *
                    FROM tool_result_cache
                    WHERE cache_key = ?
                      AND expiry_timestamp > ?
                      AND (
                        created_response_ordinal IS NULL
                        OR ? IS NULL
                        OR (? - created_response_ordinal) <= ?
                      )
                `
            )
            .get(
                cacheKey,
                now,
                currentResponseOrdinal,
                currentResponseOrdinal,
                maxResponsesAgo
            ) as Record<string, unknown> | undefined;

        return row ? this.mapCachedToolResult(row) : null;
    }

    public static getConversationResolutionContext(options: {
        guildId: string | null;
        currentChannelId: string | null;
        currentResponseOrdinal?: number | null;
        maxResponsesAgo?: number;
        now?: number;
    }): ConversationResolutionContextRecord | null {
        const now = options.now ?? Date.now();
        const row = MemoryDatabase.get()
            .prepare(
                `
                    SELECT *
                    FROM conversation_resolution_contexts
                    WHERE guild_id IS ?
                      AND expiry_timestamp > ?
                      AND (
                        created_response_ordinal IS NULL
                        OR ? IS NULL
                        OR (? - created_response_ordinal) <= ?
                      )
                    ORDER BY
                        CASE
                            WHEN channel_id = ? THEN 0
                            WHEN channel_id IS NULL THEN 2
                            ELSE 1
                        END,
                        created_timestamp DESC
                    LIMIT 1
                `
            )
            .get(
                options.guildId,
                now,
                options.currentResponseOrdinal ?? null,
                options.currentResponseOrdinal ?? null,
                options.maxResponsesAgo ?? Number.MAX_SAFE_INTEGER,
                options.currentChannelId
            ) as Record<string, unknown> | undefined;

        return row ? this.mapConversationResolutionContext(row) : null;
    }

    public static getRecentReusableGroundedContext(options: {
        guildId: string | null;
        currentChannelId: string | null;
        routeIntent?: string;
        requireSufficient?: boolean;
        currentResponseOrdinal?: number | null;
        maxResponsesAgo?: number;
        now?: number;
    }): ReusableGroundedContextRecord | null {
        const now = options.now ?? Date.now();
        const routeIntentSql = options.routeIntent ? "AND route_intent = ?" : "";
        const row = MemoryDatabase.get()
            .prepare(
                `
                    SELECT *
                    FROM reusable_grounded_contexts
                    WHERE guild_id IS ?
                      ${routeIntentSql}
                      AND expiry_timestamp > ?
                      AND (
                        created_response_ordinal IS NULL
                        OR ? IS NULL
                        OR (? - created_response_ordinal) <= ?
                      )
                      ${options.requireSufficient ? "AND sufficient = 1" : ""}
                    ORDER BY
                        CASE
                            WHEN channel_id = ? THEN 0
                            WHEN channel_id IS NULL THEN 2
                            ELSE 1
                        END,
                        created_timestamp DESC
                    LIMIT 1
                `
            )
            .get(
                ...[
                    options.guildId,
                    ...(options.routeIntent ? [options.routeIntent] : []),
                    now,
                    options.currentResponseOrdinal ?? null,
                    options.currentResponseOrdinal ?? null,
                    options.maxResponsesAgo ?? Number.MAX_SAFE_INTEGER,
                    options.currentChannelId,
                ]
            ) as Record<string, unknown> | undefined;

        return row ? this.mapReusableGroundedContext(row) : null;
    }

    private static mapReusableGroundedContext(
        row: Record<string, unknown>
    ): ReusableGroundedContextRecord {
        return {
            guildId: row.guild_id ? String(row.guild_id) : null,
            channelId: row.channel_id ? String(row.channel_id) : null,
            channelScopeKey: String(row.channel_scope_key),
            questionFingerprint: String(row.question_fingerprint),
            routeIntent: String(row.route_intent) as ReusableGroundedContextRecord["routeIntent"],
            evidenceText: String(row.evidence_text),
            citations: JSON.parse(String(row.citations_json)),
            toolRuns: JSON.parse(String(row.tool_runs_json)),
            sufficient: Boolean(Number(row.sufficient ?? 0)),
            groundingDecisionMode:
                String(row.grounding_decision_mode) as ReusableGroundedContextRecord["groundingDecisionMode"],
            createdTimestamp: Number(row.created_timestamp),
            expiryTimestamp: Number(row.expiry_timestamp),
            createdResponseOrdinal:
                row.created_response_ordinal === null ||
                row.created_response_ordinal === undefined
                    ? null
                    : Number(row.created_response_ordinal),
        };
    }

    private static mapCachedToolResult(
        row: Record<string, unknown>
    ): CachedToolResultRecord {
        return {
            cacheKey: String(row.cache_key),
            toolName: String(row.tool_name),
            guildId: row.guild_id ? String(row.guild_id) : null,
            argumentsJson: String(row.arguments_json),
            result: JSON.parse(String(row.result_json)),
            createdTimestamp: Number(row.created_timestamp),
            expiryTimestamp: Number(row.expiry_timestamp),
            createdResponseOrdinal:
                row.created_response_ordinal === null ||
                row.created_response_ordinal === undefined
                    ? null
                    : Number(row.created_response_ordinal),
        };
    }

    private static mapConversationResolutionContext(
        row: Record<string, unknown>
    ): ConversationResolutionContextRecord {
        return {
            guildId: row.guild_id ? String(row.guild_id) : null,
            channelId: row.channel_id ? String(row.channel_id) : null,
            routeIntent: String(row.route_intent) as ConversationResolutionContextRecord["routeIntent"],
            targetText: row.target_text ? String(row.target_text) : null,
            authorId: row.author_id ? String(row.author_id) : null,
            authorQuery: row.author_query ? String(row.author_query) : null,
            channelIds: JSON.parse(String(row.channel_ids_json)),
            topicText: row.topic_text ? String(row.topic_text) : null,
            channelHintText: row.channel_hint_text ? String(row.channel_hint_text) : null,
            resolvedPerson: row.resolved_person_json
                ? JSON.parse(String(row.resolved_person_json))
                : null,
            createdTimestamp: Number(row.created_timestamp),
            expiryTimestamp: Number(row.expiry_timestamp),
            createdResponseOrdinal:
                row.created_response_ordinal === null ||
                row.created_response_ordinal === undefined
                    ? null
                    : Number(row.created_response_ordinal),
        };
    }
}

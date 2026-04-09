import { MemoryDatabase } from "@/memory/MemoryDatabase";
import type {
    CachedToolResultRecord,
    ConversationResolutionContextRecord,
    ReusableGroundedContextRecord,
} from "@/memory/types";

export class CacheWriteRepository {
    public static pruneExpired(now = Date.now()): void {
        const db = MemoryDatabase.get();
        db.prepare(
            `DELETE FROM reusable_grounded_contexts WHERE expiry_timestamp <= ?`
        ).run(now);
        db.prepare(`DELETE FROM tool_result_cache WHERE expiry_timestamp <= ?`).run(now);
        db.prepare(
            `DELETE FROM conversation_resolution_contexts WHERE expiry_timestamp <= ?`
        ).run(now);
    }

    public static upsertReusableGroundedContext(
        context: ReusableGroundedContextRecord
    ): void {
        this.pruneExpired(context.createdTimestamp);

        MemoryDatabase.get()
            .prepare(
                `
                    INSERT INTO reusable_grounded_contexts (
                        guild_id,
                        channel_id,
                        channel_scope_key,
                        question_fingerprint,
                        route_intent,
                        evidence_text,
                        citations_json,
                        tool_runs_json,
                        sufficient,
                        grounding_decision_mode,
                        created_timestamp,
                        expiry_timestamp,
                        created_response_ordinal
                    ) VALUES (
                        @guildId,
                        @channelId,
                        @channelScopeKey,
                        @questionFingerprint,
                        @routeIntent,
                        @evidenceText,
                        @citationsJson,
                        @toolRunsJson,
                        @sufficient,
                        @groundingDecisionMode,
                        @createdTimestamp,
                        @expiryTimestamp,
                        @createdResponseOrdinal
                    )
                    ON CONFLICT(guild_id, channel_scope_key, question_fingerprint, route_intent)
                    DO UPDATE SET
                        channel_id = excluded.channel_id,
                        evidence_text = excluded.evidence_text,
                        citations_json = excluded.citations_json,
                        tool_runs_json = excluded.tool_runs_json,
                        sufficient = excluded.sufficient,
                        grounding_decision_mode = excluded.grounding_decision_mode,
                        created_timestamp = excluded.created_timestamp,
                        expiry_timestamp = excluded.expiry_timestamp,
                        created_response_ordinal = excluded.created_response_ordinal
                `
            )
            .run({
                guildId: context.guildId,
                channelId: context.channelId,
                channelScopeKey: context.channelScopeKey,
                questionFingerprint: context.questionFingerprint,
                routeIntent: context.routeIntent,
                evidenceText: context.evidenceText,
                citationsJson: JSON.stringify(context.citations),
                toolRunsJson: JSON.stringify(context.toolRuns),
                sufficient: context.sufficient ? 1 : 0,
                groundingDecisionMode: context.groundingDecisionMode,
                createdTimestamp: context.createdTimestamp,
                expiryTimestamp: context.expiryTimestamp,
                createdResponseOrdinal: context.createdResponseOrdinal,
            });
    }

    public static upsertCachedToolResult(entry: CachedToolResultRecord): void {
        this.pruneExpired(entry.createdTimestamp);

        MemoryDatabase.get()
            .prepare(
                `
                    INSERT INTO tool_result_cache (
                        cache_key,
                        tool_name,
                        guild_id,
                        arguments_json,
                        result_json,
                        created_timestamp,
                        expiry_timestamp,
                        created_response_ordinal
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        result_json = excluded.result_json,
                        created_timestamp = excluded.created_timestamp,
                        expiry_timestamp = excluded.expiry_timestamp,
                        created_response_ordinal = excluded.created_response_ordinal
                `
            )
            .run(
                entry.cacheKey,
                entry.toolName,
                entry.guildId,
                entry.argumentsJson,
                JSON.stringify(entry.result),
                entry.createdTimestamp,
                entry.expiryTimestamp,
                entry.createdResponseOrdinal
            );
    }

    public static upsertConversationResolutionContext(
        context: ConversationResolutionContextRecord
    ): void {
        this.pruneExpired(context.createdTimestamp);

        MemoryDatabase.get()
            .prepare(
                `
                    INSERT INTO conversation_resolution_contexts (
                        guild_id,
                        channel_scope_key,
                        channel_id,
                        route_intent,
                        target_text,
                        author_id,
                        author_query,
                        channel_ids_json,
                        topic_text,
                        channel_hint_text,
                        resolved_person_json,
                        created_timestamp,
                        expiry_timestamp,
                        created_response_ordinal
                    ) VALUES (
                        @guildId,
                        @channelScopeKey,
                        @channelId,
                        @routeIntent,
                        @targetText,
                        @authorId,
                        @authorQuery,
                        @channelIdsJson,
                        @topicText,
                        @channelHintText,
                        @resolvedPersonJson,
                        @createdTimestamp,
                        @expiryTimestamp,
                        @createdResponseOrdinal
                    )
                    ON CONFLICT(guild_id, channel_scope_key) DO UPDATE SET
                        channel_id = excluded.channel_id,
                        route_intent = excluded.route_intent,
                        target_text = excluded.target_text,
                        author_id = excluded.author_id,
                        author_query = excluded.author_query,
                        channel_ids_json = excluded.channel_ids_json,
                        topic_text = excluded.topic_text,
                        channel_hint_text = excluded.channel_hint_text,
                        resolved_person_json = excluded.resolved_person_json,
                        created_timestamp = excluded.created_timestamp,
                        expiry_timestamp = excluded.expiry_timestamp,
                        created_response_ordinal = excluded.created_response_ordinal
                `
            )
            .run({
                guildId: context.guildId,
                channelScopeKey: context.channelId || "__guild__",
                channelId: context.channelId,
                routeIntent: context.routeIntent,
                targetText: context.targetText,
                authorId: context.authorId,
                authorQuery: context.authorQuery,
                channelIdsJson: JSON.stringify(context.channelIds),
                topicText: context.topicText,
                channelHintText: context.channelHintText,
                resolvedPersonJson: context.resolvedPerson
                    ? JSON.stringify(context.resolvedPerson)
                    : null,
                createdTimestamp: context.createdTimestamp,
                expiryTimestamp: context.expiryTimestamp,
                createdResponseOrdinal: context.createdResponseOrdinal,
            });
    }
}

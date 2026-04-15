import { z } from "zod";
import { T } from "@/shared/discordTools";
import { getAppConfig } from "@/app/AppConfig";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import type {
    ActiveRetrievalSession,
    EvidenceItem,
    RetrievalSummary,
} from "@/runtime/contracts";
import type {
    DiscordToolResult,
    RetrievalMode,
    RetrievedChunk,
    SemanticContinuationCursor,
} from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

// ── Payload type ────────────────────────────────────────────────────

export type RetrievalPayload = {
    query?: string;
    mode?: RetrievalMode;
    historyMessages?: RetrievedChunk[];
    semanticMatches?: RetrievedChunk[];
    combinedResults?: RetrievedChunk[];
    sourceOrigin?: RetrievalSummary["sourceOrigin"];
    targetAuthorId?: string | null;
    targetChannelIds?: string[];
    cacheHit?: boolean;
    liveEscalated?: boolean;
    searchedChannelIds?: string[];
    fetchedChannelIds?: string[];
    cacheEnriched?: boolean;
    evidenceSufficient?: boolean;
    strongResultCount?: number;
    weakResultCount?: number;
    historyMessageCount?: number;
    semanticMatchCount?: number;
    accumulatedUniqueCount?: number;
    continuation?: {
        history?: {
            perChannelOldestMessageId?: Record<string, string | null>;
            continuationAvailable?: boolean;
        };
        semantic?: {
            cursor?: SemanticContinuationCursor | null;
            continuationAvailable?: boolean;
        };
        perChannelOldestMessageId?: Record<string, string | null>;
        continuationAvailable?: boolean;
    };
    exhaustion?: {
        historyExhaustedChannelIds?: string[];
        historyExhausted?: boolean;
        semanticExhausted?: boolean;
        exhaustedChannelIds?: string[];
        exhausted?: boolean;
    };
    beforeTimestamp?: number | null;
    afterTimestamp?: number | null;
    excludedMessageIds?: string[];
    retrievalDiagnostics?: {
        scopedEmptyRetryAttempted?: boolean;
        retryStrategy?: string;
        scopedEmptyRetryRecovered?: boolean;
    };
    scan?: {
        requestedDeepScan?: boolean;
        scanUntilExhausted?: boolean;
        targetCount?: number | null;
        pagesScanned?: number;
        maxPagesReached?: boolean;
        messageCapReached?: boolean;
        totalUniqueScanned?: number;
        totalHistoryScanned?: number;
        totalSemanticScanned?: number;
    };
};

const DEFAULT_MAX_SCAN_PAGES = 12;
const ABSOLUTE_MAX_SCAN_PAGES = 40;
const DEFAULT_MAX_RETURNED_MESSAGES = 300;
const ABSOLUTE_MAX_RETURNED_MESSAGES = 600;

// ── Strategy helpers ────────────────────────────────────────────────

function toRetrievalSummary(payload: RetrievalPayload): RetrievalSummary {
    return {
        mode: (payload.mode || "history") as RetrievalSummary["mode"],
        cacheHit: Boolean(payload.cacheHit),
        liveEscalated: Boolean(payload.liveEscalated),
        searchedChannelIds: (payload.searchedChannelIds || []).map(String),
        fetchedChannelIds: (payload.fetchedChannelIds || []).map(String),
        cacheEnriched: Boolean(payload.cacheEnriched),
        evidenceSufficient: Boolean(payload.evidenceSufficient),
        strongResultCount: Number(payload.strongResultCount || 0),
        weakResultCount: Number(payload.weakResultCount || 0),
        historyMessageCount: Number(payload.historyMessageCount || 0),
        semanticMatchCount: Number(payload.semanticMatchCount || 0),
        accumulatedUniqueCount: Number(payload.accumulatedUniqueCount || 0),
        sourceOrigin: (payload.sourceOrigin ||
            "none") as RetrievalSummary["sourceOrigin"],
        continuationAvailable: Boolean(
            payload.continuation?.continuationAvailable,
        ),
        historyContinuationAvailable:
            payload.continuation?.history?.continuationAvailable == null
                ? Boolean(payload.continuation?.continuationAvailable)
                : Boolean(
                      payload.continuation?.history?.continuationAvailable,
                  ),
        historyCursorByChannel:
            payload.continuation?.history?.perChannelOldestMessageId ||
            payload.continuation?.perChannelOldestMessageId ||
            {},
        semanticContinuationAvailable: Boolean(
            payload.continuation?.semantic?.continuationAvailable,
        ),
        semanticCursor:
            payload.continuation?.semantic?.cursor || null,
        exhaustedChannelIds: (
            payload.exhaustion?.exhaustedChannelIds || []
        ).map(String),
        historyExhausted: Boolean(payload.exhaustion?.historyExhausted),
        semanticExhausted: Boolean(payload.exhaustion?.semanticExhausted),
        beforeTimestamp:
            payload.beforeTimestamp == null
                ? null
                : Number(payload.beforeTimestamp),
        afterTimestamp:
            payload.afterTimestamp == null
                ? null
                : Number(payload.afterTimestamp),
        activeChannelIds: (payload.targetChannelIds || []).map(String),
    };
}

function toRetrievalSession(
    payload: RetrievalPayload & Record<string, unknown>,
): ActiveRetrievalSession | null {
    const targetChannelIds = Array.isArray(payload.targetChannelIds)
        ? payload.targetChannelIds.map(String)
        : [];

    if (!targetChannelIds.length) return null;

    return {
        mode: (payload.mode || "history") as RetrievalMode,
        channelIds: targetChannelIds,
        authorId:
            payload.targetAuthorId == null
                ? null
                : String(payload.targetAuthorId),
        beforeTimestamp:
            payload.beforeTimestamp == null
                ? null
                : Number(payload.beforeTimestamp),
        afterTimestamp:
            payload.afterTimestamp == null
                ? null
                : Number(payload.afterTimestamp),
        historyCursorByChannel:
            payload.continuation?.history?.perChannelOldestMessageId ||
            payload.continuation?.perChannelOldestMessageId ||
            {},
        semanticCursor:
            payload.continuation?.semantic?.cursor || null,
        seenMessageIds: (payload.combinedResults || [])
            .map((row) =>
                row && typeof row === "object" && "messageId" in row
                    ? String(row.messageId)
                    : null,
            )
            .filter((value): value is string => Boolean(value)),
        accumulatedUniqueCount: Number(
            payload.accumulatedUniqueCount || 0,
        ),
        exhaustedChannelIds: (
            payload.exhaustion?.exhaustedChannelIds || []
        ).map(String),
        historyExhausted: Boolean(payload.exhaustion?.historyExhausted),
        semanticExhausted: Boolean(payload.exhaustion?.semanticExhausted),
        continuationAvailable: Boolean(
            payload.continuation?.continuationAvailable,
        ),
    };
}

// ── Shared zod schemas ──────────────────────────────────────────────

const chunkResultSchema = z.object({
    messageId: z.string(),
    channelId: z.string(),
    channelName: z.string(),
    guildId: z.string().nullable(),
    authorId: z.string(),
    authorName: z.string(),
    content: z.string(),
    createdTimestamp: z.number(),
    jumpLink: z.string(),
    lexicalScore: z.number(),
    semanticScore: z.number(),
    recencyScore: z.number(),
    totalScore: z.number(),
});

const retrievalModeSchema = z.enum(["history", "semantic", "mixed"]);
const semanticCursorSchema = z.object({
    lastScore: z.number(),
    lastCreatedTimestamp: z.number(),
    lastMessageId: z.string(),
});

// ── Schema parameters ───────────────────────────────────────────────

const parameters = {
    type: "object",
    properties: {
        query: {
            type: "string",
            description:
                "Search query for Discord messages. Used for both lexical and semantic matching.",
        },
        channelIds: {
            type: "array",
            items: { type: "string" },
            description:
                "Limit search to these channel IDs (from resolve_channel_targets or list_guild_structure). Omit to search all indexed channels.",
        },
        authorId: {
            type: "string",
            description:
                "Filter messages to this author's Discord snowflake ID (from resolve_member_identity).",
        },
        beforeTimestamp: {
            type: "number",
            description:
                "Unix timestamp (ms). Only return messages before this time. When computing a year for a partial date (day+month only): if that day/month has already passed relative to `current_date`, use the current year; if it is still upcoming, use the previous year.",
        },
        afterTimestamp: {
            type: "number",
            description:
                "Unix timestamp (ms). Only return messages after this time. When computing a year for a partial date (day+month only): if that day/month has already passed relative to `current_date`, use the current year; if it is still upcoming, use the previous year.",
        },
        aroundMessageId: {
            type: "string",
            description:
                "A Discord message ID from previous results. Returns messages surrounding that message for reading context. Use this to zoom into a specific conversation point.",
        },
        mode: {
            type: "string",
            enum: ["history", "semantic", "mixed"],
            description:
                "Retrieval mode. 'history' = chronological order (best for reading through a channel like a file). 'semantic' = relevance-ranked search. 'mixed' = both lanes. Default: mixed.",
        },
        limit: {
            type: "number",
            description:
                "Number of messages to return per page. Default 50. Use higher values when scrolling through history.",
        },
        scanUntilExhausted: {
            type: "boolean",
            description:
                "If true, auto-paginates internally through continuation cursors until exhaustion or limits are hit. Use when the user asks to read an entire chat/history.",
        },
        targetCount: {
            type: "number",
            description:
                "Target number of unique messages to gather before stopping auto-pagination. Useful for requests like 'find 200 latest messages'.",
        },
        maxPages: {
            type: "number",
            description:
                "Maximum number of internal pages to scan in one tool call when deep scanning. Safety bound to avoid runaway scans.",
        },
        maxReturnedMessages: {
            type: "number",
            description:
                "Maximum number of messages included in this tool response payload. Scanning can continue beyond this for counting, but payload stays bounded.",
        },
        cursor: {
            type: "object",
            description:
                "Continuation cursor from a previous retrieve_messages call. Pass this exactly as received to fetch the next page and keep scrolling through history.",
            properties: {
                history: {
                    type: "object",
                    description:
                        "History pagination state. Pass through as-is from previous results.",
                    properties: {},
                    required: [],
                },
                semantic: {
                    type: "object",
                    properties: {
                        lastScore: { type: "number" },
                        lastCreatedTimestamp: { type: "number" },
                        lastMessageId: { type: "string" },
                    },
                    required: [],
                },
            },
            required: [],
        },
        excludedMessageIds: {
            type: "array",
            items: { type: "string" },
            description:
                "Message IDs to exclude from results (already seen in previous pages).",
        },
    },
    required: ["query"],
} as const;

// ── Tool definition ─────────────────────────────────────────────────

export const retrieveMessagesTool: ToolDefinition = {
    name: T.retrieve_messages,

    catalog: {
        effect: "read",
        description:
            "Search cached Discord messages first, then automatically refresh from live Discord history when needed.",
        evidenceRole: "message_evidence",
    },

    schema: {
        description:
            "Search Discord message history. Returns messages with full metadata including message IDs, author IDs, channel IDs, timestamps, and jump links. Use 'mode' to control retrieval: 'history' for chronological scrolling (like reading a file), 'semantic' for relevance-based search, 'mixed' for both. Use 'cursor' from previous results to paginate through more messages. Use 'aroundMessageId' to get context surrounding a specific message ID.",
        parameters,
    },

    capability: {
        description:
            "Read scoped Discord channel history first, add semantic matches from the same scope, and continue with live history fetches when needed.",
        inputSchema: z.object({
            query: z.string(),
            limit: z
                .number()
                .int()
                .positive()
                .describe(
                    "Page size. Omit to use the server default (typically 50). Use larger values when searching through deep history.",
                )
                .optional(),
            channelIds: z.array(z.string()).optional(),
            authorId: z.string().optional(),
            beforeTimestamp: z.number().optional(),
            afterTimestamp: z.number().optional(),
            aroundMessageId: z
                .string()
                .describe(
                    "A Discord message ID from evidence. When set, returns messages surrounding that message instead of paginated history. Use this to get context around a found message.",
                )
                .optional(),
            mode: retrievalModeSchema.optional(),
            scanUntilExhausted: z.boolean().optional(),
            targetCount: z.number().int().positive().optional(),
            maxPages: z.number().int().positive().optional(),
            maxReturnedMessages: z.number().int().positive().optional(),
            cursor: z
                .object({
                    history: z
                        .record(z.string(), z.string().nullable())
                        .optional(),
                    semantic: semanticCursorSchema.nullable().optional(),
                })
                .optional(),
            excludedMessageIds: z.array(z.string()).optional(),
        }),
        outputSchema: z.object({
            query: z.string(),
            mode: retrievalModeSchema,
            cacheHit: z.boolean(),
            liveEscalated: z.boolean(),
            searchedChannelIds: z.array(z.string()),
            fetchedChannelIds: z.array(z.string()),
            cacheEnriched: z.boolean(),
            evidenceSufficient: z.boolean(),
            strongResultCount: z.number(),
            weakResultCount: z.number(),
            historyMessageCount: z.number(),
            semanticMatchCount: z.number(),
            sourceOrigin: z.enum([
                "none",
                "cache",
                "live_refresh",
                "cache_after_refresh",
            ]),
            targetAuthorId: z.string().nullable(),
            targetChannelIds: z.array(z.string()),
            historyMessages: z.array(chunkResultSchema),
            semanticMatches: z.array(chunkResultSchema),
            combinedResults: z.array(chunkResultSchema),
            continuation: z.object({
                history: z.object({
                    perChannelOldestMessageId: z.record(
                        z.string(),
                        z.string().nullable(),
                    ),
                    continuationAvailable: z.boolean(),
                }),
                semantic: z.object({
                    cursor: semanticCursorSchema.nullable(),
                    continuationAvailable: z.boolean(),
                }),
                perChannelOldestMessageId: z.record(
                    z.string(),
                    z.string().nullable(),
                ),
                continuationAvailable: z.boolean(),
            }),
            exhaustion: z.object({
                historyExhaustedChannelIds: z.array(z.string()),
                historyExhausted: z.boolean(),
                semanticExhausted: z.boolean(),
                exhaustedChannelIds: z.array(z.string()),
                exhausted: z.boolean(),
            }),
            accumulatedWindow: z.object({
                beforeTimestamp: z.number().nullable(),
                afterTimestamp: z.number().nullable(),
            }),
            accumulatedUniqueCount: z.number(),
            beforeTimestamp: z.number().nullable(),
            afterTimestamp: z.number().nullable(),
            excludedMessageIds: z.array(z.string()),
            scan: z.object({
                requestedDeepScan: z.boolean().optional(),
                scanUntilExhausted: z.boolean().optional(),
                targetCount: z.number().nullable().optional(),
                pagesScanned: z.number().optional(),
                maxPagesReached: z.boolean().optional(),
                messageCapReached: z.boolean().optional(),
                totalUniqueScanned: z.number().optional(),
                totalHistoryScanned: z.number().optional(),
                totalSemanticScanned: z.number().optional(),
            }).optional(),
        }),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: [
            "guild context should exist for live escalation",
        ],
        postconditions: [
            "returns message evidence from the local cache after any needed live fetch",
        ],
        async run(context, args) {
            const query =
                typeof args.query === "string"
                    ? args.query
                    : context.question;
            const channelIds = Array.isArray(args.channelIds)
                ? args.channelIds.filter(
                      (value): value is string =>
                          typeof value === "string" &&
                          value.trim().length > 0,
                  )
                : undefined;
            const authorId =
                typeof args.authorId === "string" &&
                args.authorId.trim()
                    ? args.authorId.trim()
                    : undefined;
            const mode =
                typeof args.mode === "string" &&
                ["history", "semantic", "mixed"].includes(args.mode)
                    ? (args.mode as RetrievalMode)
                    : undefined;
            const cursor =
                args.cursor &&
                typeof args.cursor === "object" &&
                !Array.isArray(args.cursor)
                    ? {
                          history:
                              args.cursor.history &&
                              typeof args.cursor.history === "object" &&
                              !Array.isArray(args.cursor.history)
                                  ? Object.fromEntries(
                                        Object.entries(
                                            args.cursor.history,
                                        )
                                            .filter(
                                                ([, value]) =>
                                                    value == null ||
                                                    typeof value ===
                                                        "string",
                                            )
                                            .map(([key, value]) => [
                                                key,
                                                value == null
                                                    ? null
                                                    : String(value),
                                            ]),
                                    )
                                  : undefined,
                          semantic:
                              args.cursor.semantic &&
                              typeof args.cursor.semantic === "object" &&
                              !Array.isArray(args.cursor.semantic) &&
                              typeof args.cursor.semantic.lastScore ===
                                  "number" &&
                              typeof args.cursor.semantic
                                  .lastCreatedTimestamp === "number" &&
                              typeof args.cursor.semantic
                                  .lastMessageId === "string"
                                  ? {
                                        lastScore:
                                            args.cursor.semantic
                                                .lastScore,
                                        lastCreatedTimestamp:
                                            args.cursor.semantic
                                                .lastCreatedTimestamp,
                                        lastMessageId:
                                            args.cursor.semantic
                                                .lastMessageId,
                                    }
                                  : undefined,
                      }
                    : undefined;
            const excludedMessageIds = Array.isArray(
                args.excludedMessageIds,
            )
                ? args.excludedMessageIds.filter(
                      (value): value is string =>
                          typeof value === "string" &&
                          value.trim().length > 0,
                  )
                : undefined;
            const scanUntilExhausted = Boolean(
                args.scanUntilExhausted,
            );
            const targetCount =
                typeof args.targetCount === "number" &&
                Number.isFinite(args.targetCount) &&
                args.targetCount > 0
                    ? Math.floor(args.targetCount)
                    : null;
            const requestedMaxPages =
                typeof args.maxPages === "number" &&
                Number.isFinite(args.maxPages) &&
                args.maxPages > 0
                    ? Math.floor(args.maxPages)
                    : DEFAULT_MAX_SCAN_PAGES;
            const maxPages = Math.max(
                1,
                Math.min(requestedMaxPages, ABSOLUTE_MAX_SCAN_PAGES),
            );
            const requestedMaxReturned =
                typeof args.maxReturnedMessages === "number" &&
                Number.isFinite(args.maxReturnedMessages) &&
                args.maxReturnedMessages > 0
                    ? Math.floor(args.maxReturnedMessages)
                    : DEFAULT_MAX_RETURNED_MESSAGES;
            const maxReturnedMessages = Math.max(
                1,
                Math.min(
                    requestedMaxReturned,
                    ABSOLUTE_MAX_RETURNED_MESSAGES,
                ),
            );
            const effectiveLimit = Number(
                args.limit ||
                    getAppConfig().runtime.retrievalHistoryLimit,
            );
            const shouldDeepScan =
                scanUntilExhausted ||
                (targetCount != null && targetCount > effectiveLimit);
            let result = await UnifiedMessageRetrieval.retrieve({
                guild: context.guild,
                question: query,
                currentChannelId: context.currentChannelId,
                channelIds,
                authorId,
                mode,
                beforeTimestamp:
                    typeof args.beforeTimestamp === "number"
                        ? args.beforeTimestamp
                        : undefined,
                afterTimestamp:
                    typeof args.afterTimestamp === "number"
                        ? args.afterTimestamp
                        : undefined,
                aroundMessageId:
                    typeof args.aroundMessageId === "string" &&
                    args.aroundMessageId.trim()
                        ? args.aroundMessageId.trim()
                        : undefined,
                cursor,
                excludedMessageIds,
                limit: effectiveLimit,
                onProgress: context.onProgress,
            });
            let outputData = result as unknown as RetrievalPayload &
                Record<string, unknown>;

            if (shouldDeepScan && !args.aroundMessageId) {
                const historyById = new Map<string, RetrievedChunk>();
                const semanticById = new Map<string, RetrievedChunk>();
                const combinedById = new Map<string, RetrievedChunk>();
                const seedRows = (result.combinedResults || [])
                    .map((row) =>
                        row && typeof row === "object" ? row : null,
                    )
                    .filter((row): row is RetrievedChunk =>
                        Boolean(row),
                    );
                const seenMessageIds = new Set<string>(
                    excludedMessageIds || [],
                );
                for (const row of seedRows) {
                    const id = String(row.messageId || "");
                    if (!id) continue;
                    seenMessageIds.add(id);
                }

                const insertRows = (
                    rows: RetrievedChunk[],
                    bag: Map<string, RetrievedChunk>,
                ) => {
                    for (const row of rows) {
                        const id = String(row.messageId || "");
                        if (!id) continue;
                        if (!bag.has(id)) bag.set(id, row);
                        if (!combinedById.has(id)) combinedById.set(id, row);
                        seenMessageIds.add(id);
                    }
                };

                insertRows(
                    result.historyMessages || [],
                    historyById,
                );
                insertRows(
                    result.semanticMatches || [],
                    semanticById,
                );

                let pagesScanned = 1;
                let maxPagesReached = false;
                let exhausted = Boolean(result.exhaustion?.exhausted);
                let messageCapReached =
                    combinedById.size >= maxReturnedMessages;
                let nextCursor = {
                    history:
                        result.continuation?.history
                            ?.perChannelOldestMessageId,
                    semantic:
                        result.continuation?.semantic?.cursor || undefined,
                };

                while (
                    !exhausted &&
                    Boolean(result.continuation?.continuationAvailable) &&
                    pagesScanned < maxPages
                ) {
                    if (
                        targetCount != null &&
                        combinedById.size >= targetCount
                    ) {
                        break;
                    }
                    if (messageCapReached) {
                        break;
                    }

                    const next = await UnifiedMessageRetrieval.retrieve({
                        guild: context.guild,
                        question: query,
                        currentChannelId: context.currentChannelId,
                        channelIds,
                        authorId,
                        mode,
                        beforeTimestamp:
                            typeof args.beforeTimestamp === "number"
                                ? args.beforeTimestamp
                                : undefined,
                        afterTimestamp:
                            typeof args.afterTimestamp === "number"
                                ? args.afterTimestamp
                                : undefined,
                        cursor: nextCursor,
                        excludedMessageIds: Array.from(seenMessageIds),
                        limit: effectiveLimit,
                        onProgress: context.onProgress,
                    });

                    result = next;
                    pagesScanned += 1;
                    exhausted = Boolean(next.exhaustion?.exhausted);
                    nextCursor = {
                        history:
                            next.continuation?.history
                                ?.perChannelOldestMessageId,
                        semantic:
                            next.continuation?.semantic?.cursor ||
                            undefined,
                    };

                    insertRows(
                        next.historyMessages || [],
                        historyById,
                    );
                    insertRows(
                        next.semanticMatches || [],
                        semanticById,
                    );
                    messageCapReached =
                        combinedById.size >= maxReturnedMessages;
                }

                if (
                    pagesScanned >= maxPages &&
                    Boolean(result.continuation?.continuationAvailable) &&
                    !Boolean(result.exhaustion?.exhausted)
                ) {
                    maxPagesReached = true;
                }

                const sortByTimestamp = (
                    left: RetrievedChunk,
                    right: RetrievedChunk,
                ) =>
                    Number(left.createdTimestamp || 0) -
                    Number(right.createdTimestamp || 0);

                const historyMessages = Array.from(
                    historyById.values(),
                )
                    .sort(sortByTimestamp)
                    .slice(0, maxReturnedMessages);
                const semanticMatches = Array.from(
                    semanticById.values(),
                ).slice(0, maxReturnedMessages);
                const combinedResults = Array.from(
                    combinedById.values(),
                )
                    .sort(sortByTimestamp)
                    .slice(0, maxReturnedMessages);

                outputData = {
                    ...result,
                    historyMessages,
                    semanticMatches,
                    combinedResults,
                    historyMessageCount: historyMessages.length,
                    semanticMatchCount: semanticMatches.length,
                    accumulatedUniqueCount: combinedById.size,
                    scan: {
                        requestedDeepScan: true,
                        scanUntilExhausted,
                        targetCount,
                        pagesScanned,
                        maxPagesReached,
                        messageCapReached,
                        totalUniqueScanned: combinedById.size,
                        totalHistoryScanned: historyById.size,
                        totalSemanticScanned: semanticById.size,
                    },
                };
            }

            const qualityLabel =
                Number(outputData.historyMessageCount || 0) >= 2
                    ? "ordered history evidence"
                    : Number(outputData.historyMessageCount || 0) >= 1
                      ? "partial history evidence"
                      : Number(outputData.semanticMatchCount || 0)
                        ? "semantic evidence"
                        : "no message evidence";
            const diagnosticsSuffix =
                outputData.retrievalDiagnostics
                    ?.scopedEmptyRetryAttempted
                    ? ` (scoped retry: ${outputData.retrievalDiagnostics.retryStrategy}, recovered=${
                          outputData.retrievalDiagnostics
                              .scopedEmptyRetryRecovered
                              ? "yes"
                              : "no"
                      })`
                    : "";
            const scanSuffix = outputData.scan
                ? ` [deep scan pages=${outputData.scan.pagesScanned || 1}, unique=${outputData.scan.totalUniqueScanned || outputData.accumulatedUniqueCount}${outputData.scan.maxPagesReached ? ", max-pages" : ""}${outputData.scan.messageCapReached ? ", payload-cap" : ""}]`
                : "";

            const isDeepScan = Boolean(outputData.scan);
            const emptyReason = isDeepScan
                ? `No messages found after scanning ${outputData.scan?.pagesScanned ?? 1} page(s) — channel may be empty or not yet indexed.${diagnosticsSuffix}${scanSuffix}`
                : outputData.liveEscalated
                  ? `No messages found (live refresh attempted — channel may be empty or unindexed).${diagnosticsSuffix}${scanSuffix}`
                  : `No messages in cache yet — channel may not be indexed.${diagnosticsSuffix}${scanSuffix}`;

            return {
                tool: T.retrieve_messages,
                summary: Array.isArray(outputData.combinedResults) &&
                    outputData.combinedResults.length
                    ? `${qualityLabel}; ${outputData.historyMessageCount} history and ${outputData.semanticMatchCount} semantic result(s) ${outputData.liveEscalated ? "after refreshing Discord history" : "from cached Discord history"}.${diagnosticsSuffix}${scanSuffix}`
                    : emptyReason,
                data: outputData,
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult): EvidenceItem[] {
            if (!run.data || typeof run.data !== "object") return [];

            const payload = run.data as RetrievalPayload;
            const historyRows = payload.historyMessages || [];
            const semanticRows = payload.semanticMatches || [];
            const sourceOrigin = (payload.sourceOrigin ||
                "none") as RetrievalSummary["sourceOrigin"];

            const historyEvidence = historyRows.map((item) => ({
                tool: T.retrieve_messages,
                summary: run.summary,
                content: String(item.content || ""),
                evidenceRole: "history_evidence" as const,
                strength: "strong" as const,
                sourceOrigin,
                messageId:
                    item.messageId == null
                        ? null
                        : String(item.messageId),
                authorId:
                    item.authorId == null
                        ? null
                        : String(item.authorId),
                authorName:
                    item.authorName == null
                        ? null
                        : String(item.authorName),
                authorUsername:
                    item.authorUsername == null
                        ? null
                        : String(item.authorUsername),
                authorNickname:
                    item.authorNickname == null
                        ? null
                        : String(item.authorNickname),
                channelId:
                    item.channelId == null
                        ? null
                        : String(item.channelId),
                channelName:
                    item.channelName == null
                        ? null
                        : String(item.channelName),
                jumpLink:
                    item.jumpLink == null
                        ? null
                        : String(item.jumpLink),
                createdTimestamp:
                    item.createdTimestamp == null
                        ? null
                        : Number(item.createdTimestamp),
            }));

            const semanticEvidence = semanticRows.map((item) => {
                const body = String(item.content || "");
                const lexicalScore = Number(item.lexicalScore || 0);
                const strength: EvidenceItem["strength"] =
                    lexicalScore >= 2 ||
                    (lexicalScore >= 1 && body.length >= 80)
                        ? "strong"
                        : "weak";

                return {
                    tool: T.retrieve_messages,
                    summary: run.summary,
                    content: body,
                    evidenceRole: "semantic_evidence" as const,
                    strength,
                    sourceOrigin,
                    messageId:
                        item.messageId == null
                            ? null
                            : String(item.messageId),
                    authorId:
                        item.authorId == null
                            ? null
                            : String(item.authorId),
                    authorName:
                        item.authorName == null
                            ? null
                            : String(item.authorName),
                    authorUsername:
                        item.authorUsername == null
                            ? null
                            : String(item.authorUsername),
                    authorNickname:
                        item.authorNickname == null
                            ? null
                            : String(item.authorNickname),
                    channelId:
                        item.channelId == null
                            ? null
                            : String(item.channelId),
                    channelName:
                        item.channelName == null
                            ? null
                            : String(item.channelName),
                    jumpLink:
                        item.jumpLink == null
                            ? null
                            : String(item.jumpLink),
                    createdTimestamp:
                        item.createdTimestamp == null
                            ? null
                            : Number(item.createdTimestamp),
                };
            });

            return [...historyEvidence, ...semanticEvidence];
        },

        extractRetrievalSummary(
            run: DiscordToolResult,
        ): RetrievalSummary | null {
            if (!run.data || typeof run.data !== "object") return null;
            return toRetrievalSummary(run.data as RetrievalPayload);
        },

        extractRetrievalSession(
            run: DiscordToolResult,
        ): ActiveRetrievalSession | null {
            if (!run.data || typeof run.data !== "object") return null;
            return toRetrievalSession(
                run.data as RetrievalPayload & Record<string, unknown>,
            );
        },
    },

    display: { icon: "📨", labelPt: "Pesquisar mensagens" },
};

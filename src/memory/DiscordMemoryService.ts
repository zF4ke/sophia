import type { InArgs } from "@libsql/client";
import type { Message } from "discord.js";
import { MessageEligibility } from "@/memory/ingest/MessageEligibility";
import { MessageNormalizer } from "@/memory/ingest/MessageNormalizer";
import { MessageChunker } from "@/memory/index/MessageChunker";
import type {
    ChannelCrawlState,
    ChannelIndexState,
    HistoricalAuthorRecord,
    KnownChannelRecord,
    SearchMessageScope,
    StoredMessage,
} from "@/memory/types";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import type { ConversationTurnSummary } from "@/runtime/contracts";
import type { ChannelCandidate, RetrievedChunk } from "@/shared/appTypes";

type ChannelSummary = {
    channelId: string;
    channelName: string;
    messageCount: number;
    firstMessageTimestamp: number | null;
    lastMessageTimestamp: number | null;
    recentAuthors: string[];
};

type RuntimeRunRecord = {
    requestId: string;
    threadId: string;
    guildId: string | null;
    channelId: string | null;
    actorId: string;
    requesterDisplayName: string | null;
    trigger: string | null;
    classificationMode: string;
    runtimeMode: string;
    stopReason: string;
    confidence: string;
    question: string;
    answer: string;
    traceEvents: Array<{ label: string; detail: string; timestamp: number }>;
};

type RecentToolRunRecord = {
    requestId: string;
    toolName: string;
    argumentsJson: string;
    summary: string;
    learned: string;
    outputJson: string;
    createdTimestamp: number;
};

export type RequestNoteKind = "note" | "plan";

export type RequestNoteRecord = {
    requestId: string;
    threadId: string;
    seq: number;
    kind: RequestNoteKind;
    label: string | null;
    body: string;
    createdTimestamp: number;
};

const EMPTY_STATS = { messages: 0, chunks: 0, channels: 0 };

function now() {
    return Date.now();
}

function tokenize(query: string): string[] {
    return query
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .split(/\s+/)
        .map((term) => term.trim())
        .filter((term) => term.length > 1);
}

function buildScopeSql(
    scope: SearchMessageScope,
    alias = "m"
): { sql: string; args: InArgs } {
    const clauses: string[] = [];
    const args: InArgs = {};

    if (scope.guildId !== undefined) {
        clauses.push(`${alias}.guild_id ${scope.guildId === null ? "IS NULL" : "= :guildId"}`);
        if (scope.guildId !== null) {
            args.guildId = scope.guildId;
        }
    }

    if (scope.channelIds?.length) {
        const names = scope.channelIds.map((_, index) => `channelId${index}`);
        clauses.push(
            `${alias}.channel_id IN (${names.map((name) => `:${name}`).join(", ")})`
        );
        scope.channelIds.forEach((channelId, index) => {
            args[`channelId${index}`] = channelId;
        });
    }

    if (scope.authorIds?.length) {
        const names = scope.authorIds.map((_, index) => `authorId${index}`);
        clauses.push(
            `${alias}.author_id IN (${names.map((name) => `:${name}`).join(", ")})`
        );
        scope.authorIds.forEach((authorId, index) => {
            args[`authorId${index}`] = authorId;
        });
    }

    if (scope.beforeTimestamp != null) {
        clauses.push(`${alias}.created_timestamp < :beforeTimestamp`);
        args.beforeTimestamp = scope.beforeTimestamp;
    }

    if (scope.afterTimestamp != null) {
        clauses.push(`${alias}.created_timestamp >= :afterTimestamp`);
        args.afterTimestamp = scope.afterTimestamp;
    }

    if (scope.excludedMessageIds?.length) {
        const names = scope.excludedMessageIds.map((_, index) => `excludedMessageId${index}`);
        clauses.push(`${alias}.id NOT IN (${names.map((name) => `:${name}`).join(", ")})`);
        scope.excludedMessageIds.forEach((messageId, index) => {
            args[`excludedMessageId${index}`] = messageId;
        });
    }

    return {
        sql: clauses.length ? `WHERE ${clauses.join(" AND ")}` : "",
        args,
    };
}

function mapStoredMessage(row: Record<string, unknown>): StoredMessage {
    return {
        id: String(row.id),
        guildId: row.guild_id == null ? null : String(row.guild_id),
        channelId: String(row.channel_id),
        channelName: String(row.channel_name),
        authorId: String(row.author_id),
        authorName: String(row.author_name),
        authorUsername: row.author_username == null ? null : String(row.author_username),
        authorNickname: row.author_nickname == null ? null : String(row.author_nickname),
        content: String(row.content || ""),
        attachmentsJson: String(row.attachments_json || "[]"),
        referenceMessageId:
            row.reference_message_id == null ? null : String(row.reference_message_id),
        createdTimestamp: Number(row.created_timestamp || 0),
        jumpLink: String(row.jump_link || ""),
        isBot: Number(row.is_bot || 0),
    };
}

export class DiscordMemoryService {
    private static statsSnapshot = { ...EMPTY_STATS };
    private static indexStateSnapshot = new Map<string, ChannelIndexState>();
    private static knownChannelsSnapshot = new Map<string, KnownChannelRecord>();
    private static crawlStateSnapshot = new Map<string, ChannelCrawlState>();
    private static initialized = false;

    private static async ensureInitialized(): Promise<void> {
        await OperationalStore.initialize();
        if (!this.initialized) {
            await this.refreshSnapshots();
            this.initialized = true;
        }
    }

    private static async refreshSnapshots(): Promise<void> {
        const client = OperationalStore.getClient();
        const [statsResult, indexResult, channelsResult, crawlResult] = await Promise.all([
            client.execute(`
                SELECT
                    (SELECT COUNT(*) FROM messages) AS messages,
                    (SELECT COUNT(*) FROM message_chunks) AS chunks,
                    (SELECT COUNT(*) FROM channels) AS channels
            `),
            client.execute(
                `SELECT channel_id, last_message_id, last_indexed_timestamp FROM index_state`
            ),
            client.execute(
                `SELECT channel_id, guild_id, channel_name, channel_topic, channel_type, parent_category_id, parent_category_name, last_seen_timestamp FROM channels`
            ),
            client.execute(
                `SELECT channel_id, last_crawled_timestamp, oldest_fetched_message_id, exhausted FROM channel_crawl_state`
            ),
        ]);

        const statsRow = statsResult.rows[0] as Record<string, unknown> | undefined;
        this.statsSnapshot = statsRow
            ? {
                  messages: Number(statsRow.messages || 0),
                  chunks: Number(statsRow.chunks || 0),
                  channels: Number(statsRow.channels || 0),
              }
            : { ...EMPTY_STATS };

        this.indexStateSnapshot = new Map(
            indexResult.rows.map((row) => [
                String(row.channel_id),
                {
                    channelId: String(row.channel_id),
                    lastMessageId:
                        row.last_message_id == null ? null : String(row.last_message_id),
                    lastIndexedTimestamp:
                        row.last_indexed_timestamp == null
                            ? null
                            : Number(row.last_indexed_timestamp),
                },
            ])
        );

        this.knownChannelsSnapshot = new Map(
            channelsResult.rows.map((row) => [
                String(row.channel_id),
                {
                    channelId: String(row.channel_id),
                    guildId: row.guild_id == null ? null : String(row.guild_id),
                    channelName: String(row.channel_name),
                    channelTopic: row.channel_topic == null ? null : String(row.channel_topic),
                    channelType: row.channel_type == null ? null : String(row.channel_type),
                    parentCategoryId:
                        row.parent_category_id == null ? null : String(row.parent_category_id),
                    parentCategoryName:
                        row.parent_category_name == null ? null : String(row.parent_category_name),
                    lastSeenTimestamp: Number(row.last_seen_timestamp || 0),
                },
            ])
        );

        this.crawlStateSnapshot = new Map(
            crawlResult.rows.map((row) => [
                String(row.channel_id),
                {
                    channelId: String(row.channel_id),
                    lastCrawledTimestamp:
                        row.last_crawled_timestamp == null
                            ? null
                            : Number(row.last_crawled_timestamp),
                    oldestFetchedMessageId:
                        row.oldest_fetched_message_id == null
                            ? null
                            : String(row.oldest_fetched_message_id),
                    exhausted: Boolean(Number(row.exhausted || 0)),
                },
            ])
        );
    }

    public static async resetForTests(): Promise<void> {
        this.resetSnapshots();
        await OperationalStore.reset();
    }

    public static async resetRuntimeState(): Promise<void> {
        this.resetSnapshots();
        await OperationalStore.reset();
    }

    private static resetSnapshots(): void {
        this.initialized = false;
        this.statsSnapshot = { ...EMPTY_STATS };
        this.indexStateSnapshot.clear();
        this.knownChannelsSnapshot.clear();
        this.crawlStateSnapshot.clear();
    }

    public static isEligibleMessage(message: Message): boolean {
        return MessageEligibility.isEligible(message);
    }

    public static async ingestMessage(message: Message): Promise<void> {
        if (!this.isEligibleMessage(message)) {
            return;
        }

        if (!message.member && message.guild?.members?.fetch && message.author?.id) {
            try {
                const fetchedMember =
                    message.guild.members.cache.get(message.author.id) ||
                    (await message.guild.members.fetch(message.author.id));
                if (fetchedMember) {
                    Object.assign(message, { member: fetchedMember });
                }
            } catch {
                // Keep ingestion best-effort when member lookup is unavailable.
            }
        }

        await this.ingestStoredMessage(MessageNormalizer.toStoredMessage(message));
    }

    public static async ingestStoredMessage(stored: StoredMessage): Promise<void> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        const existingChunkCount = (
            await client.execute({
                sql: `SELECT COUNT(*) AS count FROM message_chunks WHERE message_id = :messageId`,
                args: { messageId: stored.id },
            })
        ).rows[0] as Record<string, unknown> | undefined;
        const previousChunkCount = Number(existingChunkCount?.count || 0);
        const chunks = MessageChunker.split(stored.content);

        await client.batch([
            {
                sql: `
                    INSERT INTO messages (
                        id, guild_id, channel_id, channel_name, author_id, author_name, author_username, author_nickname,
                        content, attachments_json, reference_message_id, created_timestamp,
                        jump_link, is_bot
                    ) VALUES (
                        :id, :guildId, :channelId, :channelName, :authorId, :authorName, :authorUsername, :authorNickname,
                        :content, :attachmentsJson, :referenceMessageId, :createdTimestamp,
                        :jumpLink, :isBot
                    )
                    ON CONFLICT(id) DO UPDATE SET
                        guild_id = excluded.guild_id,
                        channel_id = excluded.channel_id,
                        channel_name = excluded.channel_name,
                        author_id = excluded.author_id,
                        author_name = excluded.author_name,
                        author_username = excluded.author_username,
                        author_nickname = excluded.author_nickname,
                        content = excluded.content,
                        attachments_json = excluded.attachments_json,
                        reference_message_id = excluded.reference_message_id,
                        created_timestamp = excluded.created_timestamp,
                        jump_link = excluded.jump_link,
                        is_bot = excluded.is_bot
                `,
                args: {
                    id: stored.id,
                    guildId: stored.guildId,
                    channelId: stored.channelId,
                    channelName: stored.channelName,
                    authorId: stored.authorId,
                    authorName: stored.authorName,
                    authorUsername: stored.authorUsername ?? null,
                    authorNickname: stored.authorNickname ?? null,
                    content: stored.content,
                    attachmentsJson: stored.attachmentsJson,
                    referenceMessageId: stored.referenceMessageId,
                    createdTimestamp: stored.createdTimestamp,
                    jumpLink: stored.jumpLink,
                    isBot: stored.isBot,
                },
            },
            {
                sql: `
                    INSERT INTO channels (
                        channel_id, guild_id, channel_name, channel_topic, channel_type,
                        parent_category_id, parent_category_name, last_seen_timestamp
                    )
                    VALUES (
                        :channelId, :guildId, :channelName, :channelTopic, :channelType,
                        :parentCategoryId, :parentCategoryName, :lastSeenTimestamp
                    )
                    ON CONFLICT(channel_id) DO UPDATE SET
                        guild_id = excluded.guild_id,
                        channel_name = excluded.channel_name,
                        channel_topic = COALESCE(excluded.channel_topic, channels.channel_topic),
                        channel_type = COALESCE(excluded.channel_type, channels.channel_type),
                        parent_category_id = COALESCE(excluded.parent_category_id, channels.parent_category_id),
                        parent_category_name = COALESCE(excluded.parent_category_name, channels.parent_category_name),
                        last_seen_timestamp = excluded.last_seen_timestamp
                `,
                args: {
                    channelId: stored.channelId,
                    guildId: stored.guildId,
                    channelName: stored.channelName,
                    channelTopic: null,
                    channelType: null,
                    parentCategoryId: null,
                    parentCategoryName: null,
                    lastSeenTimestamp: stored.createdTimestamp,
                },
            },
            {
                sql: `DELETE FROM message_chunks WHERE message_id = :messageId`,
                args: { messageId: stored.id },
            },
            {
                sql: `
                    INSERT INTO index_state (channel_id, last_message_id, last_indexed_timestamp)
                    VALUES (:channelId, :lastMessageId, :lastIndexedTimestamp)
                    ON CONFLICT(channel_id) DO UPDATE SET
                        last_message_id = excluded.last_message_id,
                        last_indexed_timestamp = CASE
                            WHEN index_state.last_indexed_timestamp IS NULL THEN excluded.last_indexed_timestamp
                            WHEN excluded.last_indexed_timestamp > index_state.last_indexed_timestamp THEN excluded.last_indexed_timestamp
                            ELSE index_state.last_indexed_timestamp
                        END
                `,
                args: {
                    channelId: stored.channelId,
                    lastMessageId: stored.id,
                    lastIndexedTimestamp: stored.createdTimestamp,
                },
            },
        ]);

        for (let index = 0; index < chunks.length; index += 1) {
            await client.execute({
                sql: `
                    INSERT INTO message_chunks (
                        chunk_id, message_id, channel_id, guild_id, chunk_index, content, created_timestamp
                    ) VALUES (
                        :chunkId, :messageId, :channelId, :guildId, :chunkIndex, :content, :createdTimestamp
                    )
                `,
                args: {
                    chunkId: `${stored.id}:${index}`,
                    messageId: stored.id,
                    channelId: stored.channelId,
                    guildId: stored.guildId,
                    chunkIndex: index,
                    content: chunks[index],
                    createdTimestamp: stored.createdTimestamp,
                },
            });
        }

        this.knownChannelsSnapshot.set(stored.channelId, {
            channelId: stored.channelId,
            guildId: stored.guildId,
            channelName: stored.channelName,
            channelTopic: this.knownChannelsSnapshot.get(stored.channelId)?.channelTopic || null,
            channelType: this.knownChannelsSnapshot.get(stored.channelId)?.channelType || null,
            parentCategoryId: this.knownChannelsSnapshot.get(stored.channelId)?.parentCategoryId || null,
            parentCategoryName: this.knownChannelsSnapshot.get(stored.channelId)?.parentCategoryName || null,
            lastSeenTimestamp: stored.createdTimestamp,
        });
        this.indexStateSnapshot.set(stored.channelId, {
            channelId: stored.channelId,
            lastMessageId: stored.id,
            lastIndexedTimestamp: stored.createdTimestamp,
        });
        this.statsSnapshot.messages += 1;
        this.statsSnapshot.channels = this.knownChannelsSnapshot.size;
        this.statsSnapshot.chunks = Math.max(0, this.statsSnapshot.chunks - previousChunkCount) + chunks.length;
    }

    public static async getIndexStateAsync(channelId?: string): Promise<ChannelIndexState[]> {
        await this.ensureInitialized();
        if (!channelId) {
            return [...this.indexStateSnapshot.values()];
        }
        const state = this.indexStateSnapshot.get(channelId);
        return state ? [state] : [];
    }

    public static getIndexState(channelId?: string): ChannelIndexState[] {
        if (!channelId) {
            return [...this.indexStateSnapshot.values()];
        }
        const state = this.indexStateSnapshot.get(channelId);
        return state ? [state] : [];
    }

    public static async getChannelCrawlStateAsync(channelId?: string): Promise<ChannelCrawlState[]> {
        await this.ensureInitialized();
        if (!channelId) {
            return [...this.crawlStateSnapshot.values()];
        }
        const state = this.crawlStateSnapshot.get(channelId);
        return state ? [state] : [];
    }

    public static getChannelCrawlState(channelId?: string): ChannelCrawlState[] {
        if (!channelId) {
            return [...this.crawlStateSnapshot.values()];
        }
        const state = this.crawlStateSnapshot.get(channelId);
        return state ? [state] : [];
    }

    public static async clearAllAsync(): Promise<void> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        await client.executeMultiple(`
            DELETE FROM trace_events;
            DELETE FROM conversation_messages;
            DELETE FROM tool_runs;
            DELETE FROM runtime_runs;
            DELETE FROM message_chunks;
            DELETE FROM messages;
            DELETE FROM channels;
            DELETE FROM index_state;
            DELETE FROM channel_crawl_state;
        `);
        await this.refreshSnapshots();
    }

    public static clearAll(): void {
        void this.clearAllAsync();
    }

    public static async rebuildFtsIndex(): Promise<void> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        await client.execute(`INSERT INTO message_chunks_fts(message_chunks_fts) VALUES('rebuild')`);
    }

    public static async getStatsAsync(): Promise<{ messages: number; chunks: number; channels: number }> {
        await this.ensureInitialized();
        return { ...this.statsSnapshot };
    }

    public static getStats() {
        return { ...this.statsSnapshot };
    }

    public static async searchMessagesAsync(
        query: string,
        scope: SearchMessageScope = {},
        limit = 8
    ): Promise<RetrievedChunk[]> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        const terms = tokenize(query);
        if (!terms.length) {
            return [];
        }

        const ftsQuery = terms.map((t) => `"${t.replace(/"/g, '""')}"`).join(" OR ");
        const { sql: scopeSql, args } = buildScopeSql(scope, "m");
        (args as Record<string, unknown>).ftsQuery = ftsQuery;

        const whereClause = scopeSql
            ? scopeSql.replace("WHERE", "WHERE message_chunks_fts MATCH :ftsQuery AND")
            : "WHERE message_chunks_fts MATCH :ftsQuery";

        const rows = (
            await client.execute({
                sql: `
                    SELECT
                        mc.message_id,
                        mc.channel_id,
                        m.channel_name,
                        m.guild_id,
                        m.author_id,
                        m.author_name,
                        m.author_username,
                        m.author_nickname,
                        mc.content,
                        mc.created_timestamp,
                        m.jump_link
                    FROM message_chunks_fts fts
                    INNER JOIN message_chunks mc ON mc.rowid = fts.rowid
                    INNER JOIN messages m ON m.id = mc.message_id
                    ${whereClause}
                    ORDER BY fts.rank
                    LIMIT :ftsResultLimit
                `,
                args: { ...args, ftsResultLimit: Math.max(limit * 5, 50) },
            })
        ).rows as Array<Record<string, unknown>>;

        return rows
            .map((row) => {
                const content = String(row.content || "");
                const haystack = content.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
                const lexicalScore = terms.reduce(
                    (total, term) => total + (haystack.includes(term) ? 1 : 0),
                    0
                );
                const recencyScore = Number(row.created_timestamp || 0) / 1_000_000_000_000;
                return {
                    messageId: String(row.message_id),
                    channelId: String(row.channel_id),
                    channelName: String(row.channel_name || ""),
                    guildId: row.guild_id == null ? null : String(row.guild_id),
                    authorId: String(row.author_id),
                    authorName: String(row.author_name),
                    authorUsername: row.author_username == null ? null : String(row.author_username),
                    authorNickname: row.author_nickname == null ? null : String(row.author_nickname),
                    content,
                    createdTimestamp: Number(row.created_timestamp || 0),
                    jumpLink: String(row.jump_link || ""),
                    lexicalScore,
                    semanticScore: 0,
                    recencyScore,
                    totalScore: lexicalScore + recencyScore,
                } satisfies RetrievedChunk;
            })
            .sort(
                (left, right) =>
                    right.totalScore - left.totalScore ||
                    right.createdTimestamp - left.createdTimestamp ||
                    right.messageId.localeCompare(left.messageId)
            )
            .filter((row) => {
                const cursor = scope.semanticCursor;
                if (!cursor) {
                    return true;
                }

                if (row.totalScore < cursor.lastScore) {
                    return true;
                }
                if (row.totalScore > cursor.lastScore) {
                    return false;
                }
                if (row.createdTimestamp < cursor.lastCreatedTimestamp) {
                    return true;
                }
                if (row.createdTimestamp > cursor.lastCreatedTimestamp) {
                    return false;
                }
                return row.messageId.localeCompare(cursor.lastMessageId) < 0;
            })
            .slice(0, limit);
    }

    public static async getRecentChannelMessagesAsync(
        channelId: string,
        limit = 10
    ): Promise<StoredMessage[]> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        const rows = (
            await client.execute({
                sql: `
                    SELECT *
                    FROM messages
                    WHERE channel_id = :channelId
                    ORDER BY created_timestamp DESC
                    LIMIT :limit
                `,
                args: { channelId, limit: Math.max(1, limit) },
            })
        ).rows as Array<Record<string, unknown>>;

        return rows.map(mapStoredMessage).reverse();
    }

    public static async getRandomStoredMessageAsync(
        scope: SearchMessageScope & { channelIds: string[] },
    ): Promise<StoredMessage | null> {
        const [first] = await this.getRandomStoredMessagesAsync(scope, 1);
        return first ?? null;
    }

    public static async getRandomStoredMessagesAsync(
        scope: SearchMessageScope & { channelIds: string[] },
        limit: number,
    ): Promise<StoredMessage[]> {
        await this.ensureInitialized();
        const sanitizedLimit = Math.max(1, Math.min(100, Math.floor(limit || 1)));
        const client = OperationalStore.getClient();
        const scoped = buildScopeSql(scope);
        const result = await client.execute({
            sql: `
                SELECT *
                FROM messages m
                ${scoped.sql}
                ORDER BY RANDOM()
                LIMIT ${sanitizedLimit}
            `,
            args: scoped.args,
        });

        return (result.rows as Array<Record<string, unknown>>).map(mapStoredMessage);
    }

    public static async getChannelHistoryPageAsync(options: {
        guildId?: string | null;
        channelIds: string[];
        authorId?: string | null;
        beforeTimestamp?: number | null;
        afterTimestamp?: number | null;
        perChannelOldestMessageId?: Record<string, string | null>;
        excludedMessageIds?: string[];
        limit?: number;
        order?: "newest" | "oldest";
    }): Promise<StoredMessage[]> {
        await this.ensureInitialized();
        const limit = Math.max(1, options.limit ?? 20);
        if (!options.channelIds.length) {
            return [];
        }

        const order: "newest" | "oldest" = options.order === "oldest" ? "oldest" : "newest";
        const client = OperationalStore.getClient();
        const perChannelArgs: InArgs = {};
        const perChannelClauses = options.channelIds.map((channelId, index) => {
            const channelArg = `historyChannelId${index}`;
            perChannelArgs[channelArg] = channelId;
            const cursorMessageId = options.perChannelOldestMessageId?.[channelId];

            if (!cursorMessageId) {
                return `(m.channel_id = :${channelArg})`;
            }

            const cursorArg = `historyCursorId${index}`;
            perChannelArgs[cursorArg] = cursorMessageId;
            // Newest-first: continue with messages OLDER than cursor (id <).
            // Oldest-first: continue with messages NEWER than cursor (id >).
            const cmp = order === "oldest" ? ">" : "<";
            return `(m.channel_id = :${channelArg} AND CAST(m.id AS INTEGER) ${cmp} CAST(:${cursorArg} AS INTEGER))`;
        });

        const baseScope: SearchMessageScope = {
            guildId: options.guildId,
            authorIds: options.authorId ? [options.authorId] : undefined,
            beforeTimestamp: options.beforeTimestamp ?? undefined,
            afterTimestamp: options.afterTimestamp ?? undefined,
            excludedMessageIds: options.excludedMessageIds,
        };
        const { sql: baseScopeSql, args: baseArgs } = buildScopeSql(baseScope, "m");
        const whereParts = [
            perChannelClauses.length ? `(${perChannelClauses.join(" OR ")})` : "",
            baseScopeSql.replace(/^WHERE\s+/i, ""),
        ].filter(Boolean);

        const sqlOrder = order === "oldest" ? "ASC" : "DESC";
        const rows = (
            await client.execute({
                sql: `
                    SELECT *
                    FROM messages m
                    ${whereParts.length ? `WHERE ${whereParts.join(" AND ")}` : ""}
                    ORDER BY m.created_timestamp ${sqlOrder}
                    LIMIT :limit
                `,
                args: {
                    ...baseArgs,
                    ...perChannelArgs,
                    limit,
                },
            })
        ).rows as Array<Record<string, unknown>>;

        // For newest-first (DESC) we reverse to chronological.
        // For oldest-first (ASC) rows are already chronological.
        const mapped = rows.map(mapStoredMessage);
        return order === "oldest" ? mapped : mapped.reverse();
    }

    public static async getMessageThreadAsync(
        messageId: string,
        window = 6
    ): Promise<StoredMessage[]> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        const anchorRow = (
            await client.execute({
                sql: `SELECT * FROM messages WHERE id = :messageId`,
                args: { messageId },
            })
        ).rows[0] as Record<string, unknown> | undefined;

        if (!anchorRow) {
            return [];
        }

        const anchor = mapStoredMessage(anchorRow);
        const rows = (
            await client.execute({
                sql: `
                    SELECT *
                    FROM messages
                    WHERE channel_id = :channelId
                    ORDER BY ABS(created_timestamp - :anchorTs) ASC, created_timestamp ASC
                    LIMIT :limit
                `,
                args: {
                    channelId: anchor.channelId,
                    anchorTs: anchor.createdTimestamp,
                    limit: Math.max(1, window * 2 + 1),
                },
            })
        ).rows as Array<Record<string, unknown>>;

        return rows
            .map(mapStoredMessage)
            .sort((left, right) => left.createdTimestamp - right.createdTimestamp);
    }

    public static getMessageThread(_messageId: string, _window = 6): StoredMessage[] {
        return [];
    }

    public static async getChannelSummaryAsync(channelId: string): Promise<ChannelSummary | null> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        const summaryRow = (
            await client.execute({
                sql: `
                    SELECT
                        channel_id,
                        MAX(channel_name) AS channel_name,
                        COUNT(*) AS message_count,
                        MIN(created_timestamp) AS first_message_timestamp,
                        MAX(created_timestamp) AS last_message_timestamp
                    FROM messages
                    WHERE channel_id = :channelId
                    GROUP BY channel_id
                `,
                args: { channelId },
            })
        ).rows[0] as Record<string, unknown> | undefined;

        if (!summaryRow) {
            return null;
        }

        const authorRows = (
            await client.execute({
                sql: `
                    SELECT author_name
                    FROM messages
                    WHERE channel_id = :channelId
                    ORDER BY created_timestamp DESC
                    LIMIT 5
                `,
                args: { channelId },
            })
        ).rows as Array<Record<string, unknown>>;

        return {
            channelId: String(summaryRow.channel_id),
            channelName: String(summaryRow.channel_name || ""),
            messageCount: Number(summaryRow.message_count || 0),
            firstMessageTimestamp:
                summaryRow.first_message_timestamp == null
                    ? null
                    : Number(summaryRow.first_message_timestamp),
            lastMessageTimestamp:
                summaryRow.last_message_timestamp == null
                    ? null
                    : Number(summaryRow.last_message_timestamp),
            recentAuthors: authorRows.map((row) => String(row.author_name || "")).filter(Boolean),
        };
    }

    public static getChannelSummary(_channelId: string): ChannelSummary | null {
        return null;
    }

    public static async listRelevantChannelsAsync(
        query: string,
        scope: SearchMessageScope = {},
        limit = 6
    ): Promise<ChannelCandidate[]> {
        const results = await this.searchMessagesAsync(query, scope, 100);
        const grouped = new Map<string, ChannelCandidate>();

        for (const result of results) {
            const existing = grouped.get(result.channelId);
            if (existing) {
                existing.hitCount += 1;
                if (
                    existing.lastIndexedTimestamp == null ||
                    result.createdTimestamp > existing.lastIndexedTimestamp
                ) {
                    existing.lastIndexedTimestamp = result.createdTimestamp;
                }
                continue;
            }

            grouped.set(result.channelId, {
                channelId: result.channelId,
                channelName: result.channelName,
                hitCount: 1,
                isIndexed: true,
                matchSource: "memory",
                lastIndexedTimestamp: result.createdTimestamp,
            });
        }

        return [...grouped.values()]
            .sort(
                (left, right) =>
                    right.hitCount - left.hitCount ||
                    (right.lastIndexedTimestamp || 0) - (left.lastIndexedTimestamp || 0)
            )
            .slice(0, limit);
    }

    public static listRelevantChannels(
        _query: string,
        _scope: SearchMessageScope = {},
        _limit = 6
    ): ChannelCandidate[] {
        return [];
    }

    public static async getKnownChannelsAsync(guildId?: string | null): Promise<KnownChannelRecord[]> {
        await this.ensureInitialized();
        return [...this.knownChannelsSnapshot.values()].filter((channel) =>
            guildId === undefined ? true : channel.guildId === guildId
        );
    }

    public static getKnownChannels(guildId?: string | null) {
        return [...this.knownChannelsSnapshot.values()].filter((channel) =>
            guildId === undefined ? true : channel.guildId === guildId
        );
    }

    public static async upsertDiscoveredChannel(
        channelId: string,
        guildId: string | null,
        channelName: string,
        timestamp = now(),
        metadata?: {
            channelTopic?: string | null;
            channelType?: string | null;
            parentCategoryId?: string | null;
            parentCategoryName?: string | null;
        }
    ): Promise<void> {
        await this.ensureInitialized();
        await OperationalStore.getClient().execute({
            sql: `
                INSERT INTO channels (
                    channel_id, guild_id, channel_name, channel_topic, channel_type,
                    parent_category_id, parent_category_name, last_seen_timestamp
                )
                VALUES (
                    :channelId, :guildId, :channelName, :channelTopic, :channelType,
                    :parentCategoryId, :parentCategoryName, :lastSeenTimestamp
                )
                ON CONFLICT(channel_id) DO UPDATE SET
                    guild_id = excluded.guild_id,
                    channel_name = excluded.channel_name,
                    channel_topic = COALESCE(excluded.channel_topic, channels.channel_topic),
                    channel_type = COALESCE(excluded.channel_type, channels.channel_type),
                    parent_category_id = COALESCE(excluded.parent_category_id, channels.parent_category_id),
                    parent_category_name = COALESCE(excluded.parent_category_name, channels.parent_category_name),
                    last_seen_timestamp = excluded.last_seen_timestamp
            `,
            args: {
                channelId,
                guildId,
                channelName,
                channelTopic: metadata?.channelTopic || null,
                channelType: metadata?.channelType || null,
                parentCategoryId: metadata?.parentCategoryId || null,
                parentCategoryName: metadata?.parentCategoryName || null,
                lastSeenTimestamp: timestamp,
            },
        });

        this.knownChannelsSnapshot.set(channelId, {
            channelId,
            guildId,
            channelName,
            channelTopic: metadata?.channelTopic || this.knownChannelsSnapshot.get(channelId)?.channelTopic || null,
            channelType: metadata?.channelType || this.knownChannelsSnapshot.get(channelId)?.channelType || null,
            parentCategoryId:
                metadata?.parentCategoryId ||
                this.knownChannelsSnapshot.get(channelId)?.parentCategoryId ||
                null,
            parentCategoryName:
                metadata?.parentCategoryName ||
                this.knownChannelsSnapshot.get(channelId)?.parentCategoryName ||
                null,
            lastSeenTimestamp: timestamp,
        });
        this.statsSnapshot.channels = this.knownChannelsSnapshot.size;
    }

    public static async resolveHistoricalAuthorAsync(
        guildId: string | null,
        query: string
    ): Promise<HistoricalAuthorRecord | null> {
        await this.ensureInitialized();
        const compactQuery = query
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "")
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "");
        if (!compactQuery) {
            return null;
        }

        const rows = (
            await OperationalStore.getClient().execute({
                sql: `
                    SELECT
                        author_id,
                        author_name,
                        guild_id,
                        COUNT(*) AS message_count,
                        MAX(created_timestamp) AS last_seen_timestamp,
                        MAX(is_bot) AS is_bot
                    FROM messages
                    WHERE guild_id ${guildId === null ? "IS NULL" : "= :guildId"}
                    GROUP BY author_id, author_name, guild_id
                    ORDER BY last_seen_timestamp DESC
                `,
                args: guildId === null ? {} : { guildId },
            })
        ).rows as Array<Record<string, unknown>>;

        const normalizedRows = rows
            .map((row) => ({
                authorId: String(row.author_id),
                authorName: String(row.author_name || ""),
                guildId: row.guild_id == null ? null : String(row.guild_id),
                messageCount: Number(row.message_count || 0),
                lastSeenTimestamp: Number(row.last_seen_timestamp || 0),
                isBot: Boolean(Number(row.is_bot || 0)),
                compactName: String(row.author_name || "")
                    .normalize("NFD")
                    .replace(/[\u0300-\u036f]/g, "")
                    .toLowerCase()
                    .replace(/[^a-z0-9]+/g, ""),
            }))
            .filter((row) => row.authorId === query || row.compactName.includes(compactQuery))
            .sort((left, right) => {
                const exactLeft = left.authorId === query || left.compactName === compactQuery ? 1 : 0;
                const exactRight = right.authorId === query || right.compactName === compactQuery ? 1 : 0;
                if (exactRight !== exactLeft) {
                    return exactRight - exactLeft;
                }
                if (right.messageCount !== left.messageCount) {
                    return right.messageCount - left.messageCount;
                }
                return right.lastSeenTimestamp - left.lastSeenTimestamp;
            });

        const best = normalizedRows[0];
        if (!best) {
            return null;
        }

        return {
            authorId: best.authorId,
            authorName: best.authorName,
            guildId: best.guildId,
            messageCount: best.messageCount,
            lastSeenTimestamp: best.lastSeenTimestamp,
            isBot: best.isBot,
        };
    }

    public static async updateChannelCrawlState(
        channelId: string,
        oldestFetchedMessageId: string | null,
        exhausted: boolean
    ): Promise<void> {
        await this.ensureInitialized();
        const state: ChannelCrawlState = {
            channelId,
            lastCrawledTimestamp: now(),
            oldestFetchedMessageId,
            exhausted,
        };

        await OperationalStore.getClient().execute({
            sql: `
                INSERT INTO channel_crawl_state (
                    channel_id, last_crawled_timestamp, oldest_fetched_message_id, exhausted
                ) VALUES (
                    :channelId, :lastCrawledTimestamp, :oldestFetchedMessageId, :exhausted
                )
                ON CONFLICT(channel_id) DO UPDATE SET
                    last_crawled_timestamp = excluded.last_crawled_timestamp,
                    oldest_fetched_message_id = excluded.oldest_fetched_message_id,
                    exhausted = excluded.exhausted
            `,
            args: {
                channelId,
                lastCrawledTimestamp: state.lastCrawledTimestamp,
                oldestFetchedMessageId,
                exhausted: exhausted ? 1 : 0,
            },
        });

        this.crawlStateSnapshot.set(channelId, state);
    }

    public static async getNthHistoricalMessageAsync(
        channelId: string,
        position: number
    ): Promise<StoredMessage | null> {
        await this.ensureInitialized();
        const row = (
            await OperationalStore.getClient().execute({
                sql: `
                    SELECT *
                    FROM messages
                    WHERE channel_id = :channelId
                    ORDER BY created_timestamp ASC
                    LIMIT 1 OFFSET :offset
                `,
                args: {
                    channelId,
                    offset: Math.max(0, position - 1),
                },
            })
        ).rows[0] as Record<string, unknown> | undefined;

        return row ? mapStoredMessage(row) : null;
    }

    public static getNthHistoricalMessage(_channelId: string, _position: number) {
        return null;
    }

    public static async recordToolRun(
        requestId: string,
        guildId: string | null,
        channelId: string | null,
        userId: string,
        question: string,
        toolName: string,
        argumentsJson: string,
        summary: string,
        learned: string,
        outputJson: string,
        confidenceImproved: boolean,
        durationMs: number
    ): Promise<void> {
        await this.ensureInitialized();
        await OperationalStore.getClient().execute({
            sql: `
                INSERT INTO tool_runs (
                    request_id, guild_id, channel_id, user_id, question, tool_name, arguments_json,
                    summary, learned, output_json, confidence_improved, duration_ms, created_timestamp
                ) VALUES (
                    :requestId, :guildId, :channelId, :userId, :question, :toolName, :argumentsJson,
                    :summary, :learned, :outputJson, :confidenceImproved, :durationMs, :createdTimestamp
                )
            `,
            args: {
                requestId,
                guildId,
                channelId,
                userId,
                question,
                toolName,
                argumentsJson,
                summary,
                learned,
                outputJson,
                confidenceImproved: confidenceImproved ? 1 : 0,
                durationMs,
                createdTimestamp: now(),
            },
        });
    }

    public static async recordRuntimeRun(run: RuntimeRunRecord): Promise<void> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        await client.execute({
            sql: `
                INSERT INTO runtime_runs (
                    request_id, thread_id, guild_id, channel_id, actor_id, requester_display_name, trigger,
                    classification_mode, runtime_mode, stop_reason, confidence,
                    question, answer, created_timestamp
                ) VALUES (
                    :requestId, :threadId, :guildId, :channelId, :actorId, :requesterDisplayName, :trigger,
                    :classificationMode, :runtimeMode, :stopReason, :confidence,
                    :question, :answer, :createdTimestamp
                )
            `,
            args: {
                requestId: run.requestId,
                threadId: run.threadId,
                guildId: run.guildId,
                channelId: run.channelId,
                actorId: run.actorId,
                requesterDisplayName: run.requesterDisplayName,
                trigger: run.trigger,
                classificationMode: run.classificationMode,
                runtimeMode: run.runtimeMode,
                stopReason: run.stopReason,
                confidence: run.confidence,
                question: run.question,
                answer: run.answer,
                createdTimestamp: now(),
            },
        });

        for (const event of run.traceEvents) {
            await client.execute({
                sql: `
                    INSERT INTO trace_events (request_id, label, detail, created_timestamp)
                    VALUES (:requestId, :label, :detail, :createdTimestamp)
                `,
                args: {
                    requestId: run.requestId,
                    label: event.label,
                    detail: event.detail,
                    createdTimestamp: event.timestamp,
                },
            });
        }
    }

    public static async getRecentRuntimeRunsAsync(
        threadId: string,
        limit = 3
    ): Promise<ConversationTurnSummary[]> {
        await this.ensureInitialized();
        const rows = (
            await OperationalStore.getClient().execute({
                sql: `
                    SELECT
                        request_id,
                        requester_display_name,
                        question,
                        answer,
                        classification_mode,
                        runtime_mode,
                        stop_reason,
                        confidence,
                        created_timestamp
                    FROM runtime_runs
                    WHERE thread_id = :threadId
                    ORDER BY created_timestamp DESC
                    LIMIT :limit
                `,
                args: {
                    threadId,
                    limit: Math.max(1, limit),
                },
            })
        ).rows as Array<Record<string, unknown>>;

        return rows
            .map((row) => ({
                requestId: String(row.request_id),
                requesterDisplayName: row.requester_display_name ? String(row.requester_display_name) : null,
                question: String(row.question || ""),
                answer: String(row.answer || ""),
                classificationMode: String(row.classification_mode || ""),
                runtimeMode: String(row.runtime_mode || ""),
                stopReason: String(row.stop_reason || ""),
                confidence: String(row.confidence || ""),
                createdTimestamp: Number(row.created_timestamp || 0),
            }))
            .reverse();
    }

    public static getRecentRuntimeRuns(
        _threadId: string,
        _limit = 3
    ): ConversationTurnSummary[] {
        return [];
    }

    public static async getRecentToolRunsAsync(
        threadId: string,
        limit = 6
    ): Promise<RecentToolRunRecord[]> {
        await this.ensureInitialized();
        const rows = (
            await OperationalStore.getClient().execute({
                sql: `
                    SELECT
                        tr.request_id,
                        tr.tool_name,
                        tr.arguments_json,
                        tr.summary,
                        tr.learned,
                        tr.output_json,
                        tr.created_timestamp
                    FROM tool_runs tr
                    INNER JOIN runtime_runs rr ON rr.request_id = tr.request_id
                    WHERE rr.thread_id = :threadId
                    ORDER BY tr.created_timestamp DESC
                    LIMIT :limit
                `,
                args: {
                    threadId,
                    limit: Math.max(1, limit),
                },
            })
        ).rows as Array<Record<string, unknown>>;

        return rows.map((row) => ({
            requestId: String(row.request_id),
            toolName: String(row.tool_name),
            argumentsJson: String(row.arguments_json || "{}"),
            summary: String(row.summary || ""),
            learned: String(row.learned || ""),
            outputJson: String(row.output_json || "{}"),
            createdTimestamp: Number(row.created_timestamp || 0),
        }));
    }

    public static async addRequestNote(options: {
        requestId: string;
        threadId: string;
        kind: RequestNoteKind;
        label: string | null;
        body: string;
    }): Promise<{ seq: number }> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        const row = (
            await client.execute({
                sql: `
                    SELECT COALESCE(MAX(seq), 0) AS maxSeq
                    FROM request_notes
                    WHERE request_id = :requestId
                `,
                args: { requestId: options.requestId },
            })
        ).rows[0] as Record<string, unknown> | undefined;
        const nextSeq = Number(row?.maxSeq ?? 0) + 1;

        await client.execute({
            sql: `
                INSERT INTO request_notes (
                    request_id, thread_id, seq, kind, label, body, created_timestamp
                ) VALUES (
                    :requestId, :threadId, :seq, :kind, :label, :body, :createdTimestamp
                )
            `,
            args: {
                requestId: options.requestId,
                threadId: options.threadId,
                seq: nextSeq,
                kind: options.kind,
                label: options.label,
                body: options.body,
                createdTimestamp: now(),
            },
        });
        return { seq: nextSeq };
    }

    public static async upsertRequestPlan(options: {
        requestId: string;
        threadId: string;
        body: string;
    }): Promise<{ version: number }> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();
        const row = (
            await client.execute({
                sql: `
                    SELECT seq FROM request_notes
                    WHERE request_id = :requestId AND kind = 'plan'
                    ORDER BY seq ASC LIMIT 1
                `,
                args: { requestId: options.requestId },
            })
        ).rows[0] as Record<string, unknown> | undefined;

        if (row) {
            await client.execute({
                sql: `
                    UPDATE request_notes
                    SET body = :body, created_timestamp = :createdTimestamp
                    WHERE request_id = :requestId AND seq = :seq
                `,
                args: {
                    requestId: options.requestId,
                    seq: Number(row.seq),
                    body: options.body,
                    createdTimestamp: now(),
                },
            });
            const versionRow = (
                await client.execute({
                    sql: `
                        SELECT COUNT(*) AS versions FROM request_notes
                        WHERE request_id = :requestId AND kind = 'plan'
                    `,
                    args: { requestId: options.requestId },
                })
            ).rows[0] as Record<string, unknown> | undefined;
            const existing = Number(versionRow?.versions ?? 1);
            return { version: existing + 1 };
        }

        const { seq } = await this.addRequestNote({
            requestId: options.requestId,
            threadId: options.threadId,
            kind: "plan",
            label: null,
            body: options.body,
        });
        return { version: 1 + (seq > 0 ? 0 : 0) };
    }

    public static async getRequestPlan(requestId: string): Promise<string | null> {
        await this.ensureInitialized();
        const row = (
            await OperationalStore.getClient().execute({
                sql: `
                    SELECT body FROM request_notes
                    WHERE request_id = :requestId AND kind = 'plan'
                    ORDER BY seq ASC LIMIT 1
                `,
                args: { requestId },
            })
        ).rows[0] as Record<string, unknown> | undefined;
        return row ? String(row.body || "") : null;
    }

    public static async listRequestNotes(options: {
        requestId: string;
        threadId?: string;
        includeThreadHistory?: boolean;
        kind?: RequestNoteKind;
        label?: string;
    }): Promise<RequestNoteRecord[]> {
        await this.ensureInitialized();
        const clauses: string[] = [];
        const args: InArgs = {};

        if (options.includeThreadHistory && options.threadId) {
            clauses.push(`(request_id = :requestId OR thread_id = :threadId)`);
            args.requestId = options.requestId;
            args.threadId = options.threadId;
        } else {
            clauses.push(`request_id = :requestId`);
            args.requestId = options.requestId;
        }

        if (options.kind) {
            clauses.push(`kind = :kind`);
            args.kind = options.kind;
        }
        if (options.label) {
            clauses.push(`label = :label`);
            args.label = options.label;
        }

        const rows = (
            await OperationalStore.getClient().execute({
                sql: `
                    SELECT request_id, thread_id, seq, kind, label, body, created_timestamp
                    FROM request_notes
                    WHERE ${clauses.join(" AND ")}
                    ORDER BY created_timestamp ASC, seq ASC
                `,
                args,
            })
        ).rows as Array<Record<string, unknown>>;

        return rows.map((row) => ({
            requestId: String(row.request_id),
            threadId: String(row.thread_id),
            seq: Number(row.seq),
            kind: String(row.kind) as RequestNoteKind,
            label: row.label == null ? null : String(row.label),
            body: String(row.body || ""),
            createdTimestamp: Number(row.created_timestamp || 0),
        }));
    }

    public static async countRequestNotes(options: {
        requestId: string;
        kind?: RequestNoteKind;
    }): Promise<number> {
        await this.ensureInitialized();
        const args: InArgs = { requestId: options.requestId };
        let sql = `SELECT COUNT(*) AS n FROM request_notes WHERE request_id = :requestId`;
        if (options.kind) {
            sql += ` AND kind = :kind`;
            args.kind = options.kind;
        }
        const row = (
            await OperationalStore.getClient().execute({ sql, args })
        ).rows[0] as Record<string, unknown> | undefined;
        return Number(row?.n ?? 0);
    }

    public static async clearRequestNotes(options: {
        requestId: string;
        label?: string;
        kind?: RequestNoteKind;
    }): Promise<{ removed: number }> {
        await this.ensureInitialized();
        const clauses: string[] = [`request_id = :requestId`];
        const args: InArgs = { requestId: options.requestId };
        if (options.label) {
            clauses.push(`label = :label`);
            args.label = options.label;
        }
        if (options.kind) {
            clauses.push(`kind = :kind`);
            args.kind = options.kind;
        }
        const result = await OperationalStore.getClient().execute({
            sql: `DELETE FROM request_notes WHERE ${clauses.join(" AND ")}`,
            args,
        });
        return { removed: Number(result.rowsAffected ?? 0) };
    }

    public static async recordConversationMessages(options: {
        requestId: string;
        threadId: string;
        guildId: string | null;
        channelId: string | null;
        messageIds: string[];
    }): Promise<void> {
        await this.ensureInitialized();
        const client = OperationalStore.getClient();

        for (const messageId of options.messageIds) {
            await client.execute({
                sql: `
                    INSERT INTO conversation_messages (
                        message_id, request_id, thread_id, guild_id, channel_id, created_timestamp
                    ) VALUES (
                        :messageId, :requestId, :threadId, :guildId, :channelId, :createdTimestamp
                    )
                    ON CONFLICT(message_id) DO UPDATE SET
                        request_id = excluded.request_id,
                        thread_id = excluded.thread_id,
                        guild_id = excluded.guild_id,
                        channel_id = excluded.channel_id,
                        created_timestamp = excluded.created_timestamp
                `,
                args: {
                    messageId,
                    requestId: options.requestId,
                    threadId: options.threadId,
                    guildId: options.guildId,
                    channelId: options.channelId,
                    createdTimestamp: now(),
                },
            });
        }
    }

    public static async resolveConversationThreadIdForMessage(
        messageId: string
    ): Promise<string | null> {
        await this.ensureInitialized();
        const row = (
            await OperationalStore.getClient().execute({
                sql: `
                    SELECT thread_id
                    FROM conversation_messages
                    WHERE message_id = :messageId
                `,
                args: { messageId },
            })
        ).rows[0] as Record<string, unknown> | undefined;

        return row?.thread_id == null ? null : String(row.thread_id);
    }

    public static async repairIndexesAsync(): Promise<void> {
        await this.ensureInitialized();
    }
}

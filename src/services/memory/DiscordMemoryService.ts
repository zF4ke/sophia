import type { Message } from "discord.js";
import { MemoryDatabase } from "@/services/memory/MemoryDatabase";
import { ModelGateway } from "@/services/ai/ModelGateway";
import type { ChannelIndexState, SearchMessageScope, StoredMessage } from "@/types/discordMemory";
import type { RetrievedChunk } from "@/types/app";
import { getAppConfig } from "@/config/AppConfig";

const CHUNK_LENGTH = 500;
const EMBEDDING_BATCH_SIZE = 32;

export class DiscordMemoryService {
    public static isEligibleMessage(message: Message): boolean {
        if (!message.channel?.isTextBased()) {
            return false;
        }

        const content = message.content.trim();
        if (!content) {
            return false;
        }

        if (message.author.bot) {
            return false;
        }

        if (content.startsWith("/")) {
            return false;
        }

        if (/^`[^\n]+`$/.test(content)) {
            return false;
        }

        return true;
    }

    public static async ingestMessage(message: Message): Promise<void> {
        if (!this.isEligibleMessage(message)) {
            return;
        }

        const stored = this.toStoredMessage(message);
        await this.ingestStoredMessage(stored);
    }

    public static async ingestStoredMessage(stored: StoredMessage): Promise<void> {
        const db = MemoryDatabase.get();
        const insertMessage = db.prepare(`
            INSERT INTO messages (
                id, guild_id, channel_id, channel_name, author_id, author_name, content,
                attachments_json, reference_message_id, created_timestamp, jump_link, is_bot
            ) VALUES (
                @id, @guildId, @channelId, @channelName, @authorId, @authorName, @content,
                @attachmentsJson, @referenceMessageId, @createdTimestamp, @jumpLink, @isBot
            )
            ON CONFLICT(id) DO UPDATE SET
                channel_name = excluded.channel_name,
                author_name = excluded.author_name,
                content = excluded.content,
                attachments_json = excluded.attachments_json,
                reference_message_id = excluded.reference_message_id,
                jump_link = excluded.jump_link
        `);

        const upsertChannel = db.prepare(`
            INSERT INTO channels (channel_id, guild_id, channel_name, last_seen_timestamp)
            VALUES (@channelId, @guildId, @channelName, @createdTimestamp)
            ON CONFLICT(channel_id) DO UPDATE SET
                channel_name = excluded.channel_name,
                last_seen_timestamp = excluded.last_seen_timestamp
        `);

        const deleteChunks = db.prepare(`DELETE FROM message_chunks WHERE message_id = ?`);
        const deleteFts = db.prepare(`
            DELETE FROM chunk_fts
            WHERE chunk_id NOT IN (SELECT chunk_id FROM message_chunks)
        `);
        const deleteEmbeddings = db.prepare(`
            DELETE FROM embeddings
            WHERE chunk_id NOT IN (SELECT chunk_id FROM message_chunks)
        `);
        const insertChunk = db.prepare(`
            INSERT OR REPLACE INTO message_chunks (
                chunk_id, message_id, channel_id, guild_id, chunk_index, content, created_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `);
        const insertFts = db.prepare(`
            INSERT INTO chunk_fts (rowid, chunk_id, content)
            VALUES ((SELECT rowid FROM message_chunks WHERE chunk_id = ?), ?, ?)
        `);
        const removeFtsChunk = db.prepare(`DELETE FROM chunk_fts WHERE chunk_id = ?`);
        const insertEmbedding = db.prepare(`
            INSERT OR REPLACE INTO embeddings (chunk_id, model, embedding_json)
            VALUES (?, ?, ?)
        `);

        const chunks = this.chunkMessageContent(stored.content);

        const tx = db.transaction(() => {
            insertMessage.run(stored);
            upsertChannel.run(stored);
            deleteChunks.run(stored.id);
            for (let index = 0; index < chunks.length; index += 1) {
                const chunkId = `${stored.id}:${index}`;
                insertChunk.run(
                    chunkId,
                    stored.id,
                    stored.channelId,
                    stored.guildId,
                    index,
                    chunks[index],
                    stored.createdTimestamp
                );
                removeFtsChunk.run(chunkId);
                insertFts.run(chunkId, chunkId, chunks[index]);
            }
            deleteFts.run();
            deleteEmbeddings.run();
        });

        tx();

        const embeddingModel = this.getEmbeddingModelName();
        for (let index = 0; index < chunks.length; index += EMBEDDING_BATCH_SIZE) {
            const batch = chunks.slice(index, index + EMBEDDING_BATCH_SIZE);
            const vectors = await ModelGateway.embedTexts(batch);
            vectors.forEach((vector, offset) => {
                const chunkIndex = index + offset;
                const chunkId = `${stored.id}:${chunkIndex}`;
                insertEmbedding.run(chunkId, embeddingModel, JSON.stringify(vector));
            });
        }

        this.updateIndexState(stored.channelId, stored.id, stored.createdTimestamp);
    }

    public static updateIndexState(
        channelId: string,
        lastMessageId: string,
        lastIndexedTimestamp: number
    ): void {
        const db = MemoryDatabase.get();
        db.prepare(`
            INSERT INTO index_state (channel_id, last_message_id, last_indexed_timestamp)
            VALUES (?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                last_message_id = excluded.last_message_id,
                last_indexed_timestamp = excluded.last_indexed_timestamp
        `).run(channelId, lastMessageId, lastIndexedTimestamp);
    }

    public static getIndexState(channelId?: string): ChannelIndexState[] {
        const db = MemoryDatabase.get();
        const rows = channelId
            ? db.prepare(`SELECT * FROM index_state WHERE channel_id = ?`).all(channelId)
            : db.prepare(`SELECT * FROM index_state ORDER BY channel_id`).all();

        return rows.map((row: any) => ({
            channelId: row.channel_id as string,
            lastMessageId: (row.last_message_id as string | null) || null,
            lastIndexedTimestamp: (row.last_indexed_timestamp as number | null) || null,
        }));
    }

    public static clearAll(): void {
        const db = MemoryDatabase.get();
        db.exec(`
            DELETE FROM embeddings;
            DELETE FROM chunk_fts;
            DELETE FROM message_chunks;
            DELETE FROM messages;
            DELETE FROM channels;
            DELETE FROM index_state;
            DELETE FROM tool_runs;
        `);
    }

    public static getStats(): {
        messages: number;
        chunks: number;
        channels: number;
    } {
        const db = MemoryDatabase.get();
        const messages = db.prepare(`SELECT COUNT(*) AS count FROM messages`).get() as { count: number };
        const chunks = db.prepare(`SELECT COUNT(*) AS count FROM message_chunks`).get() as { count: number };
        const channels = db.prepare(`SELECT COUNT(*) AS count FROM channels`).get() as { count: number };

        return {
            messages: messages.count,
            chunks: chunks.count,
            channels: channels.count,
        };
    }

    public static async searchMessagesAsync(
        query: string,
        scope: SearchMessageScope = {},
        limit = 8
    ): Promise<RetrievedChunk[]> {
        const db = MemoryDatabase.get();
        const normalizedQuery = query.trim();
        if (!normalizedQuery) {
            return [];
        }

        const scopeClause = this.buildScopeClause(scope);
        const lexicalRows = db
            .prepare(`
                SELECT
                    mc.chunk_id,
                    mc.message_id,
                    mc.channel_id,
                    m.channel_name,
                    mc.guild_id,
                    m.author_id,
                    m.author_name,
                    mc.content,
                    mc.created_timestamp,
                    m.jump_link,
                    bm25(chunk_fts) AS lexical_rank
                FROM chunk_fts
                JOIN message_chunks mc ON chunk_fts.chunk_id = mc.chunk_id
                JOIN messages m ON m.id = mc.message_id
                WHERE chunk_fts MATCH ?
                ${scopeClause.sql}
                ORDER BY lexical_rank
                LIMIT ?
            `)
            .all(this.toFtsQuery(normalizedQuery), ...scopeClause.params, Math.max(limit * 5, 20)) as Array<Record<string, unknown>>;

        if (!lexicalRows.length) {
            return [];
        }

        const [queryVector] = await ModelGateway.embedTexts([normalizedQuery]);
        const candidates = lexicalRows.map((row) => {
            const embeddingRow = db
                .prepare(`SELECT embedding_json FROM embeddings WHERE chunk_id = ?`)
                .get(row.chunk_id) as { embedding_json?: string } | undefined;
            const embedding = embeddingRow?.embedding_json
                ? (JSON.parse(embeddingRow.embedding_json) as number[])
                : [];
            const semanticScore = embedding.length ? this.cosineSimilarity(queryVector, embedding) : 0;
            const lexicalScore = Math.max(0.05, 1 / (1 + Math.abs(Number(row.lexical_rank || 0))));
            const ageMs = Date.now() - Number(row.created_timestamp);
            const recencyScore = 1 / (1 + ageMs / (1000 * 60 * 60 * 24 * 30));
            const totalScore = lexicalScore * 0.45 + semanticScore * 0.45 + recencyScore * 0.1;

            return {
                messageId: String(row.message_id),
                channelId: String(row.channel_id),
                channelName: String(row.channel_name),
                guildId: row.guild_id ? String(row.guild_id) : null,
                authorId: String(row.author_id),
                authorName: String(row.author_name),
                content: String(row.content),
                createdTimestamp: Number(row.created_timestamp),
                jumpLink: String(row.jump_link),
                lexicalScore,
                semanticScore,
                recencyScore,
                totalScore,
            } satisfies RetrievedChunk;
        });

        return candidates
            .sort((a, b) => b.totalScore - a.totalScore)
            .slice(0, limit);
    }

    public static getMessageThread(messageId: string, window = 6): StoredMessage[] {
        const db = MemoryDatabase.get();
        const anchor = db.prepare(`SELECT * FROM messages WHERE id = ?`).get(messageId) as Record<string, unknown> | undefined;
        if (!anchor) {
            return [];
        }

        const rows = db
            .prepare(`
                SELECT *
                FROM messages
                WHERE channel_id = ?
                  AND created_timestamp BETWEEN ? AND ?
                ORDER BY created_timestamp ASC
            `)
            .all(
                anchor.channel_id,
                Number(anchor.created_timestamp) - window * 60_000,
                Number(anchor.created_timestamp) + window * 60_000
            ) as Array<Record<string, unknown>>;

        return rows.map(this.mapStoredMessage);
    }

    public static getChannelSummary(channelId: string): {
        channelId: string;
        messageCount: number;
        recentMessages: StoredMessage[];
    } | null {
        const db = MemoryDatabase.get();
        const countRow = db.prepare(`SELECT COUNT(*) AS count FROM messages WHERE channel_id = ?`).get(channelId) as { count: number };
        if (!countRow.count) {
            return null;
        }

        const rows = db
            .prepare(`
                SELECT *
                FROM messages
                WHERE channel_id = ?
                ORDER BY created_timestamp DESC
                LIMIT 8
            `)
            .all(channelId) as Array<Record<string, unknown>>;

        return {
            channelId,
            messageCount: countRow.count,
            recentMessages: rows.map(this.mapStoredMessage),
        };
    }

    public static listRelevantChannels(
        query: string,
        scope: SearchMessageScope = {},
        limit = 6
    ): Array<{ channelId: string; channelName: string; hitCount: number }> {
        const db = MemoryDatabase.get();
        const scopeClause = this.buildScopeClause(scope);
        const rows = db
            .prepare(`
                SELECT
                    mc.channel_id,
                    m.channel_name,
                    COUNT(*) AS hit_count
                FROM chunk_fts
                JOIN message_chunks mc ON chunk_fts.chunk_id = mc.chunk_id
                JOIN messages m ON m.id = mc.message_id
                WHERE chunk_fts MATCH ?
                ${scopeClause.sql}
                GROUP BY mc.channel_id, m.channel_name
                ORDER BY hit_count DESC, m.channel_name ASC
                LIMIT ?
            `)
            .all(this.toFtsQuery(query), ...scopeClause.params, limit) as Array<Record<string, unknown>>;

        return rows.map((row) => ({
            channelId: String(row.channel_id),
            channelName: String(row.channel_name),
            hitCount: Number(row.hit_count),
        }));
    }

    public static getNthHistoricalMessage(channelId: string, position: number): StoredMessage | null {
        const db = MemoryDatabase.get();
        const row = db
            .prepare(`
                SELECT *
                FROM messages
                WHERE channel_id = ?
                ORDER BY created_timestamp ASC
                LIMIT 1 OFFSET ?
            `)
            .get(channelId, Math.max(0, position - 1)) as Record<string, unknown> | undefined;

        return row ? this.mapStoredMessage(row) : null;
    }

    public static recordToolRun(
        guildId: string | null,
        channelId: string | null,
        userId: string,
        question: string,
        toolName: string,
        summary: string
    ): void {
        const db = MemoryDatabase.get();
        db.prepare(`
            INSERT INTO tool_runs (
                guild_id, channel_id, user_id, question, tool_name, summary, created_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `).run(guildId, channelId, userId, question, toolName, summary, Date.now());
    }

    public static repairIndexes(): void {
        const db = MemoryDatabase.get();
        db.exec(`
            INSERT INTO chunk_fts(rowid, chunk_id, content)
            SELECT rowid, chunk_id, content
            FROM message_chunks
            WHERE chunk_id NOT IN (SELECT chunk_id FROM chunk_fts);
        `);
    }

    private static getEmbeddingModelName(): string {
        return getAppConfig().modelProfile.embeddingModel;
    }

    private static toStoredMessage(message: Message): StoredMessage {
        return {
            id: message.id,
            guildId: message.guildId || null,
            channelId: message.channelId,
            channelName: "name" in message.channel ? message.channel.name || message.channelId : message.channelId,
            authorId: message.author.id,
            authorName: message.member?.displayName || message.author.username,
            content: message.content.trim(),
            attachmentsJson: JSON.stringify(
                message.attachments.map((attachment) => ({
                    id: attachment.id,
                    name: attachment.name,
                    url: attachment.url,
                    contentType: attachment.contentType,
                }))
            ),
            referenceMessageId: message.reference?.messageId || null,
            createdTimestamp: message.createdTimestamp,
            jumpLink: message.url,
            isBot: message.author.bot ? 1 : 0,
        };
    }

    private static chunkMessageContent(content: string): string[] {
        if (content.length <= CHUNK_LENGTH) {
            return [content];
        }

        const chunks: string[] = [];
        let currentIndex = 0;
        while (currentIndex < content.length) {
            chunks.push(content.slice(currentIndex, currentIndex + CHUNK_LENGTH));
            currentIndex += CHUNK_LENGTH;
        }
        return chunks;
    }

    private static buildScopeClause(scope: SearchMessageScope): {
        sql: string;
        params: string[];
    } {
        const clauses: string[] = [];
        const params: string[] = [];

        if (scope.guildId) {
            clauses.push(`AND mc.guild_id = ?`);
            params.push(scope.guildId);
        }

        if (scope.channelIds?.length) {
            clauses.push(
                `AND mc.channel_id IN (${scope.channelIds.map(() => "?").join(", ")})`
            );
            params.push(...scope.channelIds);
        }

        return {
            sql: clauses.join("\n"),
            params,
        };
    }

    private static mapStoredMessage(row: Record<string, unknown>): StoredMessage {
        return {
            id: String(row.id),
            guildId: row.guild_id ? String(row.guild_id) : null,
            channelId: String(row.channel_id),
            channelName: String(row.channel_name),
            authorId: String(row.author_id),
            authorName: String(row.author_name),
            content: String(row.content),
            attachmentsJson: String(row.attachments_json),
            referenceMessageId: row.reference_message_id ? String(row.reference_message_id) : null,
            createdTimestamp: Number(row.created_timestamp),
            jumpLink: String(row.jump_link),
            isBot: Number(row.is_bot),
        };
    }

    private static toFtsQuery(query: string): string {
        return query
            .split(/\s+/)
            .map((term) => term.replace(/["']/g, "").trim())
            .filter((term) => term.length > 1)
            .join(" OR ");
    }

    private static cosineSimilarity(a: number[], b: number[]): number {
        if (!a.length || !b.length || a.length !== b.length) {
            return 0;
        }

        let dot = 0;
        let magnitudeA = 0;
        let magnitudeB = 0;

        for (let index = 0; index < a.length; index += 1) {
            dot += a[index] * b[index];
            magnitudeA += a[index] * a[index];
            magnitudeB += b[index] * b[index];
        }

        if (!magnitudeA || !magnitudeB) {
            return 0;
        }

        return dot / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
    }
}

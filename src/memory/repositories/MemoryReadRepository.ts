import { MemoryDatabase } from "@/memory/MemoryDatabase";
import type { ChannelIndexState, StoredMessage } from "@/memory/types";

export class MemoryReadRepository {
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

    public static getStats(): { messages: number; chunks: number; channels: number } {
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

    public static getEmbedding(chunkId: string): number[] {
        const row = MemoryDatabase.get()
            .prepare(`SELECT embedding_json FROM embeddings WHERE chunk_id = ?`)
            .get(chunkId) as { embedding_json?: string } | undefined;

        return row?.embedding_json ? (JSON.parse(row.embedding_json) as number[]) : [];
    }

    public static getMessageThread(messageId: string, window = 6): StoredMessage[] {
        const db = MemoryDatabase.get();
        const anchor = db.prepare(`SELECT * FROM messages WHERE id = ?`).get(messageId) as Record<
            string,
            unknown
        > | undefined;
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
        const countRow = db.prepare(`SELECT COUNT(*) AS count FROM messages WHERE channel_id = ?`).get(channelId) as {
            count: number;
        };
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
        whereSql: string,
        params: string[],
        limit = 6
    ): Array<{ channelId: string; channelName: string; hitCount: number }> {
        const rows = MemoryDatabase.get()
            .prepare(`
                SELECT
                    mc.channel_id,
                    m.channel_name,
                    COUNT(*) AS hit_count
                FROM chunk_fts
                JOIN message_chunks mc ON chunk_fts.chunk_id = mc.chunk_id
                JOIN messages m ON m.id = mc.message_id
                WHERE chunk_fts MATCH ?
                ${whereSql}
                GROUP BY mc.channel_id, m.channel_name
                ORDER BY hit_count DESC, m.channel_name ASC
                LIMIT ?
            `)
            .all(query, ...params, limit) as Array<Record<string, unknown>>;

        return rows.map((row) => ({
            channelId: String(row.channel_id),
            channelName: String(row.channel_name),
            hitCount: Number(row.hit_count),
        }));
    }

    public static getNthHistoricalMessage(channelId: string, position: number): StoredMessage | null {
        const row = MemoryDatabase.get()
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
}

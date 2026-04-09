import { MemoryDatabase } from "@/memory/MemoryDatabase";
import type { StoredMessage } from "@/memory/types";

export class MemoryWriteRepository {
    public static upsertMessageWithChunks(stored: StoredMessage, chunks: string[]): void {
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

        db.transaction(() => {
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
        })();
    }

    public static upsertEmbeddings(
        messageId: string,
        offset: number,
        model: string,
        vectors: number[][]
    ): void {
        const db = MemoryDatabase.get();
        const insertEmbedding = db.prepare(`
            INSERT OR REPLACE INTO embeddings (chunk_id, model, embedding_json)
            VALUES (?, ?, ?)
        `);

        vectors.forEach((vector, index) => {
            const chunkId = `${messageId}:${offset + index}`;
            insertEmbedding.run(chunkId, model, JSON.stringify(vector));
        });
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

    public static clearAll(): void {
        MemoryDatabase.get().exec(`
            DELETE FROM embeddings;
            DELETE FROM chunk_fts;
            DELETE FROM message_chunks;
            DELETE FROM messages;
            DELETE FROM channels;
            DELETE FROM index_state;
            DELETE FROM tool_runs;
        `);
    }

    public static repairIndexes(): void {
        MemoryDatabase.get().exec(`
            INSERT INTO chunk_fts(rowid, chunk_id, content)
            SELECT rowid, chunk_id, content
            FROM message_chunks
            WHERE chunk_id NOT IN (SELECT chunk_id FROM chunk_fts);
        `);
    }

    public static recordToolRun(
        guildId: string | null,
        channelId: string | null,
        userId: string,
        question: string,
        toolName: string,
        summary: string
    ): void {
        MemoryDatabase.get()
            .prepare(`
                INSERT INTO tool_runs (
                    guild_id, channel_id, user_id, question, tool_name, summary, created_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            `)
            .run(guildId, channelId, userId, question, toolName, summary, Date.now());
    }
}

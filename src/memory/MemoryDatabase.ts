import Database from "better-sqlite3";
import path from "path";
import { FileSystemService } from "@/platform/storage/FileSystemService";

export class MemoryDatabase {
    private static db: Database.Database | null = null;

    public static get(): Database.Database {
        if (!this.db) {
            const configuredPath = process.env.SOPHIA_MEMORY_DB_PATH;
            const dbPath =
                configuredPath ||
                path.join(FileSystemService.getDir("memory"), "discord-memory.sqlite");
            this.db = new Database(dbPath);
            this.db.pragma("journal_mode = WAL");
            this.db.pragma("foreign_keys = ON");
            this.initialize(this.db);
        }

        return this.db;
    }

    public static reset(): void {
        if (this.db) {
            this.db.close();
            this.db = null;
        }
    }

    private static initialize(db: Database.Database): void {
        db.exec(`
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                guild_id TEXT,
                channel_name TEXT NOT NULL,
                last_seen_timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                guild_id TEXT,
                channel_id TEXT NOT NULL,
                channel_name TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_name TEXT NOT NULL,
                content TEXT NOT NULL,
                attachments_json TEXT NOT NULL,
                reference_message_id TEXT,
                created_timestamp INTEGER NOT NULL,
                jump_link TEXT NOT NULL,
                is_bot INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_messages_channel_time
            ON messages(channel_id, created_timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_guild_time
            ON messages(guild_id, created_timestamp DESC);

            CREATE TABLE IF NOT EXISTS message_chunks (
                chunk_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                guild_id TEXT,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_timestamp INTEGER NOT NULL,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_message_chunks_message
            ON message_chunks(message_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
            USING fts5(
                chunk_id UNINDEXED,
                content,
                tokenize = 'porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES message_chunks(chunk_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS index_state (
                channel_id TEXT PRIMARY KEY,
                last_message_id TEXT,
                last_indexed_timestamp INTEGER
            );

            CREATE TABLE IF NOT EXISTS tool_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT,
                channel_id TEXT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_timestamp INTEGER NOT NULL
            );
        `);
    }
}

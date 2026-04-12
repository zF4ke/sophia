import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";
import { createClient, type Client, type InArgs } from "@libsql/client";
import { getAppConfig } from "@/app/AppConfig";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import { OPERATIONAL_SCHEMA_VERSION } from "@/runtime/storage/schema";

async function wait(ms: number): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, ms));
}

async function deletePathWithRetries(target: string): Promise<void> {
    for (let attempt = 0; attempt < 5; attempt += 1) {
        if (!fs.existsSync(target)) {
            return;
        }

        try {
            fs.rmSync(target, { force: true });
            return;
        } catch (error) {
            const code = error && typeof error === "object" && "code" in error ? String((error as NodeJS.ErrnoException).code) : "";
            if ((code === "EBUSY" || code === "EPERM") && attempt < 4) {
                await wait(50 * (attempt + 1));
                continue;
            }
            throw error;
        }
    }
}

async function deleteSqliteFamily(filePath: string): Promise<void> {
    for (const suffix of ["", "-wal", "-shm"]) {
        await deletePathWithRetries(`${filePath}${suffix}`);
    }
}

async function querySingleValue(
    client: Client,
    sql: string,
    args: InArgs = {}
): Promise<string | null> {
    const row = (await client.execute({ sql, args })).rows[0] as Record<string, unknown> | undefined;
    if (!row) {
        return null;
    }

    const firstKey = Object.keys(row)[0];
    const value = firstKey ? row[firstKey] : null;
    return value == null ? null : String(value);
}

export class OperationalStore {
    private static client: Client | null = null;
    private static initPromise: Promise<void> | null = null;

    public static async initialize(): Promise<void> {
        if (!this.initPromise) {
            this.initPromise = this.initializeInternal();
        }

        await this.initPromise;
    }

    public static getClient(): Client {
        if (!this.client) {
            const dbPath = getAppConfig().runtime.operationalDbPath;
            FileSystemService.ensureDirectoryExists(path.dirname(dbPath));
            this.client = createClient({
                url: pathToFileURL(dbPath).toString(),
            });
        }

        return this.client;
    }

    public static async reset(): Promise<void> {
        if (this.client) {
            this.client.close();
            this.client = null;
        }
        this.initPromise = null;
    }

    private static async initializeInternal(): Promise<void> {
        const dbPath = getAppConfig().runtime.operationalDbPath;
        FileSystemService.ensureDirectoryExists(path.dirname(dbPath));

        let client = this.getClient();
        const shouldReset = await this.shouldResetExistingDatabase(client, dbPath);
        if (shouldReset) {
            const resetInPlace = await this.resetDatabaseContents(client);
            if (!resetInPlace) {
                client.close();
                this.client = null;
                await deleteSqliteFamily(dbPath);
                client = this.getClient();
            }
        }

        await this.initializeSchema(client);
    }

    private static async shouldResetExistingDatabase(
        client: Client,
        dbPath: string
    ): Promise<boolean> {
        if (!fs.existsSync(dbPath)) {
            return false;
        }

        const tables = (
            await client.execute(`
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            `)
        ).rows as Array<Record<string, unknown>>;

        if (!tables.length) {
            return false;
        }

        const tableNames = new Set(tables.map((row) => String(row.name)));
        if (!tableNames.has("runtime_metadata") || !tableNames.has("runtime_runs")) {
            return true;
        }

        const version = await querySingleValue(
            client,
            `SELECT value FROM runtime_metadata WHERE key = 'operational_schema_version'`
        );
        if (version !== OPERATIONAL_SCHEMA_VERSION) {
            return true;
        }

        const runtimeRunColumns = (
            await client.execute(`PRAGMA table_info(runtime_runs)`)
        ).rows as Array<Record<string, unknown>>;
        const columnNames = new Set(runtimeRunColumns.map((row) => String(row.name)));
        return !columnNames.has("trigger");
    }

    private static async resetDatabaseContents(client: Client): Promise<boolean> {
        try {
            const tables = (
                await client.execute(`
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                `)
            ).rows as Array<Record<string, unknown>>;

            await client.execute("PRAGMA foreign_keys = OFF");
            for (const row of tables) {
                const tableName = String(row.name || "");
                if (!tableName) {
                    continue;
                }
                const safeName = tableName.replace(/"/g, '""');
                await client.execute(`DROP TABLE IF EXISTS \"${safeName}\"`);
            }
            await client.execute("PRAGMA foreign_keys = ON");
            return true;
        } catch {
            return false;
        }
    }

    private static async initializeSchema(client: Client): Promise<void> {
        await client.executeMultiple(`
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS runtime_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                guild_id TEXT,
                channel_name TEXT NOT NULL,
                channel_type TEXT,
                parent_category_id TEXT,
                parent_category_name TEXT,
                last_seen_timestamp INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_channels_guild
            ON channels(guild_id, channel_id);

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                guild_id TEXT,
                channel_id TEXT NOT NULL,
                channel_name TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_name TEXT NOT NULL,
                author_username TEXT,
                author_nickname TEXT,
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

            CREATE INDEX IF NOT EXISTS idx_messages_author_time
            ON messages(author_id, created_timestamp DESC);

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

            CREATE INDEX IF NOT EXISTS idx_message_chunks_channel_time
            ON message_chunks(channel_id, created_timestamp DESC);

            CREATE TABLE IF NOT EXISTS index_state (
                channel_id TEXT PRIMARY KEY,
                last_message_id TEXT,
                last_indexed_timestamp INTEGER
            );

            CREATE TABLE IF NOT EXISTS channel_crawl_state (
                channel_id TEXT PRIMARY KEY,
                last_crawled_timestamp INTEGER,
                oldest_fetched_message_id TEXT,
                exhausted INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS tool_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                guild_id TEXT,
                channel_id TEXT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                learned TEXT NOT NULL,
                output_json TEXT NOT NULL,
                confidence_improved INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                created_timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_runs (
                request_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                guild_id TEXT,
                channel_id TEXT,
                actor_id TEXT NOT NULL,
                trigger TEXT,
                classification_mode TEXT NOT NULL,
                runtime_mode TEXT NOT NULL,
                stop_reason TEXT NOT NULL,
                confidence TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                label TEXT NOT NULL,
                detail TEXT NOT NULL,
                created_timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                guild_id TEXT,
                channel_id TEXT,
                created_timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                guild_id TEXT,
                canonical_name TEXT NOT NULL,
                created_timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                subject_entity_id TEXT,
                predicate TEXT NOT NULL,
                object_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                scope TEXT NOT NULL,
                stability TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                subject_entity_id TEXT,
                predicate TEXT NOT NULL,
                canonical_value TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                first_confirmed_at INTEGER,
                last_confirmed_at INTEGER,
                evidence_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                from_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                to_entity_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                updated_timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                guild_id TEXT,
                channel_id TEXT,
                summary TEXT NOT NULL,
                occurred_at INTEGER NOT NULL,
                created_timestamp INTEGER NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS message_chunks_fts
            USING fts5(content, content='message_chunks', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
        `);

        await client.execute(`
            CREATE TRIGGER IF NOT EXISTS message_chunks_fts_ai AFTER INSERT ON message_chunks BEGIN
                INSERT INTO message_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        `);
        await client.execute(`
            CREATE TRIGGER IF NOT EXISTS message_chunks_fts_ad AFTER DELETE ON message_chunks BEGIN
                INSERT INTO message_chunks_fts(message_chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
            END
        `);
        await client.execute(`
            CREATE TRIGGER IF NOT EXISTS message_chunks_fts_au AFTER UPDATE ON message_chunks BEGIN
                INSERT INTO message_chunks_fts(message_chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                INSERT INTO message_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        `);

        await client.execute({
            sql: `
                INSERT INTO runtime_metadata (key, value)
                VALUES ('operational_schema_version', :value)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            `,
            args: { value: OPERATIONAL_SCHEMA_VERSION },
        });
    }
}

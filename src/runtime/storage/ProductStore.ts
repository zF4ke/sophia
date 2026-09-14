import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type { Client, InValue } from "@libsql/client";
import { AppPaths } from "@/app/AppPaths";
import { MigrationBackup } from "@/shared/storage/MigrationBackup";

/** Durable user-created cards and procedures, independent of the message index. */
export class ProductStore {
    private static client: Client | null = null;
    private static opening: Promise<void> | null = null;
    static async initialize() {
        if (!this.opening) this.opening = this.open().catch(error => { this.client?.close(); this.client = null; this.opening = null; throw error; });
        await this.opening;
    }
    private static async open() {
        fs.mkdirSync(AppPaths.storageRoot, { recursive: true });
        const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
        this.client = createClient({ url: pathToFileURL(path.join(AppPaths.storageRoot, "products.sqlite")).toString() });
        await MigrationBackup.database(this.client, path.join(AppPaths.storageRoot, "products.sqlite"));
        await this.client.executeMultiple(`
            CREATE TABLE IF NOT EXISTS migrations(source TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS artifact_owners(message_id TEXT PRIMARY KEY,owner_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS artifact_sources(message_id TEXT NOT NULL,source TEXT NOT NULL,PRIMARY KEY(message_id,source));
            CREATE TABLE IF NOT EXISTS artifact_revisions(message_id TEXT NOT NULL,revision INTEGER NOT NULL,spec_json TEXT,game_state TEXT,created_at INTEGER NOT NULL,PRIMARY KEY(message_id,revision));
            CREATE TABLE IF NOT EXISTS artifacts(message_id TEXT PRIMARY KEY,channel_id TEXT NOT NULL,guild_id TEXT,expires_at INTEGER,spec_json TEXT,view_section INTEGER NOT NULL DEFAULT 0,game_state TEXT,created_timestamp INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_artifacts_expiry ON artifacts(expires_at);
            CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY,guild_id TEXT,name TEXT NOT NULL,description TEXT NOT NULL,steps_json TEXT NOT NULL,created_by TEXT,created_timestamp INTEGER NOT NULL,updated_timestamp INTEGER NOT NULL);
        `);
    }
    static getClient(): Client { if (!this.client) throw new Error("Product store not initialized."); return this.client; }
    static async migrateLegacy(filename: string) {
        await this.initialize();
        const source = path.resolve(filename);
        if (!fs.existsSync(source) || (await this.client!.execute({ sql: "SELECT source FROM migrations WHERE source=?", args: [source] })).rows.length) return;
        const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
        const legacy = createClient({ url: pathToFileURL(source).toString() });
        const statements: Array<{ sql: string; args: InValue[] }> = [];
        try {
            await MigrationBackup.database(legacy, source);
            for (const [table, defaults] of Object.entries({
                artifacts: { message_id: null, channel_id: "", guild_id: null, expires_at: null, spec_json: null, view_section: 0, game_state: null, created_timestamp: 0 },
                workflows: { id: null, guild_id: null, name: "", description: "", steps_json: "[]", created_by: null, created_timestamp: 0, updated_timestamp: 0 },
            })) {
                const exists = await legacy.execute({ sql: "SELECT name FROM sqlite_master WHERE type='table' AND name=?", args: [table] });
                if (!exists.rows.length) continue;
                const columns = Object.keys(defaults);
                const rows = (await legacy.execute(`SELECT * FROM ${table}`)).rows;
                for (const row of rows) statements.push({ sql: `INSERT OR IGNORE INTO ${table}(${columns.join(",")}) VALUES(${columns.map(() => "?").join(",")})`, args: columns.map(column => (row[column] ?? defaults[column as keyof typeof defaults]) as InValue) });
            }
            statements.push({ sql: "INSERT OR IGNORE INTO migrations(source) VALUES(?)", args: [source] });
            await this.client!.batch(statements, "write");
        } finally { legacy.close(); }
    }
}

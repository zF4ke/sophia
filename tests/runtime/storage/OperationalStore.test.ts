import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";
import { createClient } from "@libsql/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { OPERATIONAL_SCHEMA_VERSION } from "@/runtime/storage/schema";

function uniqueRuntimeDir(): string {
    return path.join(
        process.cwd(),
        "storage",
        "test-runtime-store",
        `${Date.now()}-${Math.random().toString(16).slice(2)}`
    );
}

describe("OperationalStore", () => {
    let runtimeDir = "";
    let operationalDbPath = "";
    let checkpointDbPath = "";

    beforeEach(() => {
        runtimeDir = uniqueRuntimeDir();
        operationalDbPath = path.join(runtimeDir, "operational.sqlite");
        checkpointDbPath = path.join(runtimeDir, "checkpoints.sqlite");
        process.env.RUNTIME_OPERATIONAL_DB_PATH = operationalDbPath;
        process.env.RUNTIME_CHECKPOINT_DB_PATH = checkpointDbPath;
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
    });

    afterEach(async () => {
        await OperationalStore.reset();
        try {
            fs.rmSync(runtimeDir, { recursive: true, force: true });
        } catch {
            // Windows file handles can linger briefly after reset; the test uses unique dirs.
        }
    });

    it("rebuilds incompatible operational storage schemas on startup", async () => {
        fs.mkdirSync(runtimeDir, { recursive: true });
        const client = createClient({ url: pathToFileURL(operationalDbPath).toString() });
        await client.executeMultiple(`
            CREATE TABLE runtime_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO runtime_metadata (key, value)
            VALUES ('operational_schema_version', 'stale-version');

            CREATE TABLE runtime_runs (
                request_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                guild_id TEXT
            );
        `);
        client.close();

        await OperationalStore.initialize();

        const tables = await OperationalStore.getClient().execute(`PRAGMA table_info(runtime_runs)`);
        const columns = new Set(
            (tables.rows as Array<Record<string, unknown>>).map((row) => String(row.name))
        );
        const metadata = await OperationalStore.getClient().execute(
            `SELECT value FROM runtime_metadata WHERE key = 'operational_schema_version'`
        );

        expect(columns.has("trigger")).toBe(true);
        expect(columns.has("classification_mode")).toBe(true);
        expect(columns.has("answer")).toBe(true);
        expect(String((metadata.rows[0] as Record<string, unknown>).value)).toBe(
            OPERATIONAL_SCHEMA_VERSION
        );
    });
});


import { canonicalSource } from "@/shared/sourceReference";
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";
import type { Client } from "@libsql/client";
import { AppPaths } from "@/app/AppPaths";
import { MigrationBackup } from "@/shared/storage/MigrationBackup";
import { buildPrefixFtsQuery, tokenizeQueryTerms } from "@/shared/textSearch";

export interface MemoryAudience { actorId: string; guildId: string | null; channelId: string | null; privateResponse?: boolean }
export type MemoryScope = "channel" | "guild" | "user" | "preference";
export function validatePortablePreference(key: string, value: string): void {
    const valid = key === "response_length" ? ["brief", "adaptive", "detailed"].includes(value)
        : key === "tone" ? ["balanced", "casual", "formal"].includes(value)
        : key === "language" ? /^[a-z]{2,3}(?:-[A-Z]{2})?$/.test(value) : false;
    if (!valid) throw new Error("Portable preferences support response_length (brief/adaptive/detailed), tone (balanced/casual/formal), or language (a language code such as pt or en-US).");
}
export interface KnowledgeMemory {
    id: string; key: string; value: string; scope: string; ownerId: string | null;
    guildId: string | null; channelId: string | null; revision: number; sources: string[]; updatedAt: number;
}

/** One identity and knowledge collection. Audience metadata controls disclosure. */
export class KnowledgeStore {
    private client: Client | null = null;
    private opening: Promise<void> | null = null;
    constructor(private readonly filename = path.join(AppPaths.storageRoot, "knowledge.sqlite")) {}
    async initialize(): Promise<void> {
        if (!this.opening) this.opening = this.open().catch(error => { this.client?.close(); this.client = null; this.opening = null; throw error; });
        await this.opening;
    }
    private async open() {
        fs.mkdirSync(path.dirname(this.filename), { recursive: true });
        const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
        const client = this.client = createClient({ url: pathToFileURL(this.filename).toString() });
        await MigrationBackup.database(client, this.filename, "v5-preferences");
        await client.batch([
            "CREATE TABLE IF NOT EXISTS invalid_sources(source TEXT PRIMARY KEY,invalidated_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS deleted_message_sources(source TEXT PRIMARY KEY)",
            "CREATE TABLE IF NOT EXISTS identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT NOT NULL, name TEXT NOT NULL)",
            { sql: "INSERT OR IGNORE INTO identity VALUES(1,?,'Sophia')", args: [randomUUID()] },
            `CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, owner_id TEXT, guild_id TEXT, channel_id TEXT,
                scope TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
                sources_json TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)`,
            "CREATE INDEX IF NOT EXISTS memory_audience ON memories(guild_id,channel_id,owner_id,deleted)",
            "CREATE UNIQUE INDEX IF NOT EXISTS portable_preference ON memories(owner_id,key) WHERE scope='preference' AND deleted=0",
            "CREATE TABLE IF NOT EXISTS knowledge_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            "UPDATE memories SET scope='quarantine' WHERE scope='guild' AND owner_id IS NULL AND channel_id IS NULL AND sources_json='[]'",
            `CREATE TABLE IF NOT EXISTS dream_jobs (id TEXT PRIMARY KEY, input_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0, error TEXT, created_at INTEGER NOT NULL)`,
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(key,value,content='memories',content_rowid='rowid',tokenize='unicode61 remove_diacritics 2')",
            `CREATE TRIGGER IF NOT EXISTS memories_insert AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid,key,value) VALUES(new.rowid,new.key,new.value); END`,
            `CREATE TRIGGER IF NOT EXISTS memories_update AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts,rowid,key,value) VALUES('delete',old.rowid,old.key,old.value);
                INSERT INTO memories_fts(rowid,key,value) VALUES(new.rowid,new.key,new.value); END`,
        ], "write");
        await client.execute("UPDATE dream_jobs SET status='pending' WHERE status='running'");
    }
    async identity() {
        await this.initialize();
        const row = (await this.client!.execute("SELECT id,name FROM identity WHERE singleton=1")).rows[0];
        return { id: String(row.id), name: String(row.name) };
    }
    async forgetTask(taskId: string, actorId: string): Promise<void> {
        await this.initialize();
        await this.client!.execute({ sql: "UPDATE dream_jobs SET status='done',input_json=json_set(input_json,'$.question','','$.answer','','$.toolEvidence',json('[]')),error='task_forgotten' WHERE json_extract(input_json,'$.taskId')=? AND json_extract(input_json,'$.audience.actorId')=?", args: [taskId, actorId] });
    }
    async preferences(actorId: string): Promise<Record<string, string>> {
        await this.initialize();
        if (!actorId) return {};
        const rows = (await this.client!.execute({ sql: "SELECT key,value FROM memories WHERE scope='preference' AND owner_id=? AND deleted=0 ORDER BY key", args: [actorId] })).rows;
        const result: Record<string, string> = {};
        for (const row of rows) {
            const key = String(row.key), value = String(row.value);
            validatePortablePreference(key, value);
            result[key] = value;
        }
        return result;
    }
    async isSourceInvalid(source: string): Promise<boolean> {
        source = canonicalSource(source);
        await this.initialize();
        return (await this.client!.execute({ sql: "SELECT source FROM invalid_sources WHERE source=? UNION ALL SELECT source FROM deleted_message_sources WHERE source=? OR instr(?,source||'#')=1", args: [source, source, source] })).rows.length > 0;
    }
    async invalidateSource(source: string, allVersions = true): Promise<void> {
        source = canonicalSource(source);
        await this.initialize();
        await this.client!.batch([
            ...(allVersions ? [{ sql: "INSERT OR IGNORE INTO deleted_message_sources VALUES(?)", args: [source.split("#")[0]] }] : []),
            { sql: "INSERT OR IGNORE INTO invalid_sources VALUES(?,?)", args: [source, Date.now()] },
            { sql: `UPDATE memories SET value='',deleted=1,revision=revision+1,updated_at=? WHERE deleted=0 AND EXISTS (SELECT 1 FROM json_each(memories.sources_json) WHERE value=? OR (?=1 AND instr(value,?||'#')=1))`, args: [Date.now(), source, allVersions ? 1 : 0, source.split("#")[0]] },
            { sql: `UPDATE dream_jobs SET status='done',input_json=json_set(input_json,'$.question','','$.answer','','$.toolEvidence',json('[]')),error='source_changed' WHERE EXISTS (SELECT 1 FROM json_each(dream_jobs.input_json,'$.sources') WHERE value=? OR (?=1 AND instr(value,?||'#')=1))`, args: [source, allVersions ? 1 : 0, source.split("#")[0]] },
        ], "write");
    }
    async dreamContext(a: MemoryAudience) {
        await this.initialize();
        const privateResponse = !a.guildId || a.privateResponse ? 1 : 0;
        const rows = (await this.client!.execute({ sql: "SELECT key,value,deleted FROM memories WHERE owner_id=? AND ((guild_id IS ? AND channel_id IS ?) OR scope='preference' OR (scope='user' AND ?=1)) AND (scope<>'user' OR ?=1) ORDER BY updated_at DESC LIMIT 200", args: [a.actorId, a.guildId, a.channelId, privateResponse, privateResponse] })).rows;
        return { existing: rows.filter(row => !row.deleted).map(row => ({ key: String(row.key), value: String(row.value) })), suppressedLabels: rows.filter(row => row.deleted).map(row => String(row.key)) };
    }
    async migrateLegacy(filename: string): Promise<void> {
        if (!fs.existsSync(filename)) return;
        await this.initialize();
        const marker = `legacy:${path.resolve(filename)}`;
        if ((await this.client!.execute({ sql: "SELECT key FROM knowledge_metadata WHERE key=?", args: [marker] })).rows.length) return;
        const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
        const legacy = createClient({ url: pathToFileURL(filename).toString() });
        try {
            await MigrationBackup.database(legacy, filename);
            if (!(await legacy.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='long_term_memories'")).rows.length) return;
            const rows = (await legacy.execute("SELECT * FROM long_term_memories")).rows;
            await this.client!.batch([
                ...rows.map(row => ({ sql: `INSERT OR IGNORE INTO memories(id,owner_id,guild_id,channel_id,scope,key,value,sources_json,created_at,updated_at)
                    VALUES(?,?,?,NULL,?,?,?,'[]',?,?)`, args: [String(row.id), row.user_id == null ? null : String(row.user_id),
                        row.guild_id == null ? null : String(row.guild_id), "quarantine",
                        String(row.key), String(row.value), Number(row.created_timestamp), Number(row.updated_timestamp)] })),
                { sql: "INSERT OR IGNORE INTO knowledge_metadata VALUES(?,?)", args: [marker, String(Date.now())] },
            ], "write");
        } finally { legacy.close(); }
    }
    /** Caller must enforce operator authority; quarantine never enters recall. */
    async quarantine(guildId: string | null, offset = 0) {
        await this.initialize();
        if (!Number.isSafeInteger(offset) || offset < 0) throw new Error("Invalid quarantine offset.");
        const rows = (await this.client!.execute({ sql: "SELECT id,key,value,revision FROM memories WHERE scope='quarantine' AND deleted=0 AND guild_id IS ? ORDER BY id LIMIT 50 OFFSET ?", args: [guildId, offset] })).rows;
        return rows.map(row => ({ id: String(row.id), key: String(row.key), value: String(row.value), revision: Number(row.revision) }));
    }
    async adoptQuarantined(audience: MemoryAudience, id: string, revision: number): Promise<void> {
        await this.initialize();
        if (!audience.actorId) throw new Error("Memory requires an authenticated owner.");
        const result = await this.client!.execute({ sql: "UPDATE memories SET scope='user',owner_id=?,channel_id=?,revision=revision+1,updated_at=? WHERE id=? AND revision=? AND scope='quarantine' AND deleted=0 AND guild_id IS ?",
            args: [audience.actorId, audience.channelId, Date.now(), id, revision, audience.guildId] });
        if (result.rowsAffected !== 1) throw new Error("Quarantined memory is unavailable or its revision changed.");
    }
    async remember(audience: MemoryAudience, input: { key: string; value: string; scope: MemoryScope; sources?: string[] }): Promise<KnowledgeMemory> {
        await this.initialize();
        if (input.scope === "preference") {
            validatePortablePreference(input.key, input.value);
            audience = { ...audience, guildId: null, channelId: null };
            if ((await this.client!.execute({ sql: "SELECT id FROM memories WHERE scope='preference' AND owner_id=? AND key=? AND deleted=0", args: [audience.actorId, input.key] })).rows.length) throw new Error("This preference already exists. Use memory_update with its ID and revision.");
        }
        if (!audience.actorId) throw new Error("Memory requires an authenticated owner.");
        if (input.scope === "channel" && !audience.channelId) throw new Error("Channel memory requires a channel.");
        if (input.scope === "guild" && !audience.guildId) throw new Error("Guild memory requires a guild.");
        const forgotten = await this.client!.execute({ sql: "SELECT id FROM memories WHERE owner_id=? AND guild_id IS ? AND channel_id IS ? AND key=? AND deleted=1",
            args: [audience.actorId, audience.guildId, audience.channelId, input.key] });
        if (forgotten.rows.length) throw new Error("This memory was forgotten. It cannot be recreated automatically.");
        const id = randomUUID();
        const timestamp = Date.now();
        const inserted = await this.client!.execute({ sql: `INSERT INTO memories(id,owner_id,guild_id,channel_id,scope,key,value,sources_json,created_at,updated_at)
            SELECT ?,?,?,?,?,?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM invalid_sources JOIN json_each(?) ON source=value OR EXISTS(SELECT 1 FROM deleted_message_sources d WHERE d.source=value OR instr(value,d.source||'#')=1))
            AND NOT EXISTS (SELECT 1 FROM memories WHERE owner_id=? AND guild_id IS ? AND channel_id IS ? AND key=? AND deleted=1)`, args: [id, audience.actorId, audience.guildId, audience.channelId, input.scope, input.key, input.value,
                JSON.stringify((input.sources ?? []).map(canonicalSource)), timestamp, timestamp, JSON.stringify((input.sources ?? []).map(canonicalSource)), audience.actorId, audience.guildId, audience.channelId, input.key] });
        if (inserted.rowsAffected !== 1) throw new Error("The source was deleted or this memory was forgotten.");
        return { id, ownerId: audience.actorId, guildId: audience.guildId, channelId: audience.channelId, scope: input.scope,
            key: input.key, value: input.value, revision: 1, sources: input.sources ?? [], updatedAt: timestamp };
    }
    async search(audience: MemoryAudience, query = "", limit = 8): Promise<KnowledgeMemory[]> {
        await this.initialize();
        if (!audience.actorId) return [];
        if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Invalid memory result limit.");
        const terms = tokenizeQueryTerms(query);
        // Private owner memory follows private replies. Guild memory is shared.
        const sql = `SELECT m.* FROM memories m ${terms.length ? "JOIN memories_fts ON memories_fts.rowid=m.rowid" : ""}
            WHERE m.deleted=0 AND (
                (m.scope='guild' AND m.guild_id=:guild AND :guild IS NOT NULL)
                OR (m.scope='channel' AND m.guild_id IS :guild AND m.channel_id=:channel AND (:guild IS NOT NULL OR m.owner_id=:actor))
                OR (m.scope='user' AND m.owner_id=:actor AND :private=1)
                OR (m.scope='preference' AND m.owner_id=:actor)
            ) ${terms.length ? "AND memories_fts MATCH :query" : ""}
            ORDER BY ${terms.length ? "memories_fts.rank," : ""} m.updated_at DESC,m.rowid DESC LIMIT :limit`;
        const result = await this.client!.execute({ sql, args: { guild: audience.guildId, channel: audience.channelId, private: !audience.guildId || audience.privateResponse ? 1 : 0,
            actor: audience.actorId, limit, ...(terms.length ? { query: buildPrefixFtsQuery(terms) } : {}) } });
        return result.rows.map(row => ({ id: String(row.id), key: String(row.key), value: String(row.value), scope: String(row.scope),
            ownerId: row.owner_id == null ? null : String(row.owner_id), guildId: row.guild_id == null ? null : String(row.guild_id),
            channelId: row.channel_id == null ? null : String(row.channel_id), revision: Number(row.revision),
            sources: row.scope === "preference" ? [] : JSON.parse(String(row.sources_json)) as string[], updatedAt: Number(row.updated_at) }));
    }
    async ownedMetadata(id: string, actorId: string) {
        await this.initialize();
        const row = (await this.client!.execute({ sql: "SELECT scope,guild_id,channel_id,sources_json FROM memories WHERE id=? AND owner_id=? AND deleted=0", args: [id, actorId] })).rows[0];
        return row ? { scope: String(row.scope), guildId: row.guild_id === null ? null : String(row.guild_id), channelId: row.channel_id === null ? null : String(row.channel_id), sources: JSON.parse(String(row.sources_json)) as string[] } : null;
    }
    async revise(audience: MemoryAudience, id: string, revision: number, value: string | null, sources?: string[]): Promise<void> {
        await this.initialize();
        const current = (await this.client!.execute({ sql: "SELECT scope,key FROM memories WHERE id=? AND owner_id=?", args: [id, audience.actorId] })).rows[0];
        if (current?.scope === "preference" && value !== null) validatePortablePreference(String(current.key), value);
        // Owner and exact revision are required. A tombstone cannot be edited back to life.
        const result = await this.client!.execute({ sql: `UPDATE memories SET value=?,deleted=?,revision=revision+1,updated_at=?,sources_json=COALESCE(?,sources_json)
            WHERE id=? AND owner_id=? AND revision=? AND deleted=0 AND NOT EXISTS(SELECT 1 FROM invalid_sources JOIN json_each(?) ON source=value OR EXISTS(SELECT 1 FROM deleted_message_sources d WHERE d.source=value OR instr(value,d.source||'#')=1))`,
            args: [value ?? "", value === null ? 1 : 0, Date.now(), sources ? JSON.stringify(sources) : null, id, audience.actorId, revision, JSON.stringify(sources ?? [])] });
        if (result.rowsAffected !== 1) throw new Error("Memory is unavailable, belongs to another owner, or its revision changed.");
    }
    async close() { if (this.opening) await this.opening; this.client?.close(); this.client = null; this.opening = null; }

    async enqueueDream(id: string, input: DreamInput): Promise<void> {
        input = { ...input, sources: input.sources.map(canonicalSource) };
        await this.initialize();
        await this.client!.execute({ sql: "INSERT OR IGNORE INTO dream_jobs(id,input_json,created_at) SELECT ?,?,? WHERE NOT EXISTS (SELECT 1 FROM invalid_sources JOIN json_each(?) ON source=value OR EXISTS(SELECT 1 FROM deleted_message_sources d WHERE d.source=value OR instr(value,d.source||'#')=1))", args: [id, JSON.stringify(input), Date.now(), JSON.stringify(input.sources.map(canonicalSource))] });
    }
    async nextDream(): Promise<{ id: string; input: DreamInput } | null> {
        await this.initialize();
        const results = await this.client!.batch([
            "UPDATE dream_jobs SET status='running',attempts=attempts+1 WHERE id=(SELECT id FROM dream_jobs WHERE status='pending' AND attempts<3 ORDER BY created_at LIMIT 1) RETURNING id,input_json",
        ], "write");
        const row = results[0].rows[0];
        return row ? { id: String(row.id), input: JSON.parse(String(row.input_json)) as DreamInput } : null;
    }
    async finishDream(id: string, memories: Array<{ key: string; value: string; scope?: "channel" | "preference" }>, error?: string): Promise<void> {
        await this.initialize();
        const job = (await this.client!.execute({ sql: "SELECT input_json,attempts FROM dream_jobs WHERE id=? AND status='running'", args: [id] })).rows[0];
        if (!job) return; // A deleted source can invalidate an in-flight dream.
        if (error) {
            await this.client!.execute({ sql: "UPDATE dream_jobs SET status=?,error=? WHERE id=? AND status='running'", args: [Number(job.attempts) >= 3 ? "failed" : "pending", error, id] });
            return;
        }
        const input = JSON.parse(String(job.input_json)) as DreamInput;
        const a = input.audience;
        const timestamp = Date.now();
        for (const memory of memories) if (memory.scope === "preference") validatePortablePreference(memory.key, memory.value);
        await this.client!.batch([
            ...memories.map((memory, index) => {
                const destination = memory.scope === "preference" ? { ...a, guildId: null, channelId: null } : a;
                return { sql: `INSERT OR IGNORE INTO memories(id,owner_id,guild_id,channel_id,scope,key,value,sources_json,created_at,updated_at)
                SELECT ?,?,?,?,?,?,?,?,?,? WHERE NOT EXISTS (
                    SELECT 1 FROM memories WHERE owner_id=? AND guild_id IS ? AND channel_id IS ? AND key=?)
                    AND EXISTS (SELECT 1 FROM dream_jobs WHERE id=? AND status='running')
                    AND NOT EXISTS (SELECT 1 FROM invalid_sources JOIN json_each(?) ON source=value OR EXISTS(SELECT 1 FROM deleted_message_sources d WHERE d.source=value OR instr(value,d.source||'#')=1))`,
                args: [`dream:${id}:${index}`, a.actorId, destination.guildId, destination.channelId, memory.scope === "preference" ? "preference" : a.privateResponse ? "user" : "channel", memory.key, memory.value, JSON.stringify(input.sources.map(canonicalSource)), timestamp, timestamp,
                    a.actorId, destination.guildId, destination.channelId, memory.key, id, JSON.stringify(input.sources.map(canonicalSource))] }; }),
            { sql: "UPDATE dream_jobs SET status='done',error=NULL WHERE id=? AND status='running'", args: [id] },
        ], "write");
    }
}

export interface DreamInput { audience: MemoryAudience; question: string; answer: string; sources: string[];
    taskId?: string; toolEvidence?: Array<{ tool: string; summary: string; succeeded: boolean }> }

export const knowledgeStore = new KnowledgeStore();

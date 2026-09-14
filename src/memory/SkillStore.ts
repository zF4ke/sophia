import { randomUUID, createHash } from "node:crypto";
import { ProductStore } from "@/runtime/storage/ProductStore";
import { DISCORD_TOOL_NAMES } from "@/shared/discordTools";
import type { MemoryAudience } from "./KnowledgeStore";
import { knowledgeStore } from "./KnowledgeStore";

export interface SkillBody {
    sourceLinks?: string[];
    privateOnly?: boolean;
    provenance?: { dreamId: string; taskId: string; sources: string[] };
    name: string; description: string; instructions: string; capabilities: string[];
    examples: string[]; status: "draft" | "ready";
    steps?: Array<{ tool: string; args: Record<string, unknown>; label?: string }>;
}
export interface SavedSkill extends SkillBody { id: string; revision: number; ownerId: string | null; scope: string }
export function skillContentHash(body: SkillBody): string {
    return createHash("sha256").update(JSON.stringify({ name: body.name, description: body.description, instructions: body.instructions, capabilities: body.capabilities, examples: body.examples, steps: body.steps ?? [] })).digest("hex");
}
const audienceSql = `deleted=0 AND ((owner_id=:actor AND guild_id IS :guild AND channel_id IS :channel)
    OR (scope='guild' AND guild_id=:guild AND :guild IS NOT NULL AND status='ready'))`;
const bindings = (a: MemoryAudience) => ({ actor: a.actorId, guild: a.guildId, channel: a.channelId });
export class SkillStore {
    private static async sourceAvailable(body: SkillBody): Promise<boolean> {
        for (const source of body.provenance?.sources ?? []) if (await knowledgeStore.isSourceInvalid(source)) return false;
        return true;
    }

    static async draftFromDream(a: MemoryAudience, body: SkillBody, provenance: NonNullable<SkillBody["provenance"]>): Promise<void> {
        if (!a.actorId || !a.channelId || !await this.sourceAvailable({ ...body, provenance })) return;
        if (body.capabilities.some(name => !(DISCORD_TOOL_NAMES as readonly string[]).includes(name))) throw new Error("Draft names an unknown capability.");
        await this.initialize();
        const id = `dream:${provenance.dreamId}`;
        const draft = JSON.stringify({ ...body, status: "draft", provenance, privateOnly: Boolean(a.privateResponse) });
        const now = Date.now();
        await ProductStore.getClient().batch([
            { sql: `INSERT OR IGNORE INTO skills(id,owner_id,guild_id,channel_id,scope,status,name,revision,body_json,created_at,updated_at)
                SELECT ?,?,?,?,'channel','draft',?,1,?,?,? WHERE NOT EXISTS(SELECT 1 FROM skills WHERE owner_id=? AND guild_id IS ? AND channel_id IS ? AND lower(name)=lower(?))
                AND NOT EXISTS(SELECT 1 FROM forgotten_skill_tasks WHERE task_id=? AND owner_id=?)`,
                args: [id, a.actorId, a.guildId, a.channelId, body.name, draft, now, now, a.actorId, a.guildId, a.channelId, body.name, provenance.taskId, a.actorId] },
            { sql: "INSERT OR IGNORE INTO skill_revisions SELECT ?,1,?,? WHERE changes()>0", args: [id, draft, now] },
        ], "write");
    }
    static async initialize() {
        await ProductStore.initialize();
        const c = ProductStore.getClient();
        await c.execute("CREATE TABLE IF NOT EXISTS forgotten_skill_tasks(task_id TEXT NOT NULL,owner_id TEXT NOT NULL,PRIMARY KEY(task_id,owner_id))");
        await c.executeMultiple(`CREATE TABLE IF NOT EXISTS skills(id TEXT PRIMARY KEY,owner_id TEXT,guild_id TEXT,channel_id TEXT,scope TEXT NOT NULL,status TEXT NOT NULL,
            name TEXT NOT NULL,revision INTEGER NOT NULL,body_json TEXT NOT NULL,deleted INTEGER NOT NULL DEFAULT 0,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS skill_revisions(skill_id TEXT NOT NULL,revision INTEGER NOT NULL,body_json TEXT NOT NULL,created_at INTEGER NOT NULL,PRIMARY KEY(skill_id,revision));
            CREATE INDEX IF NOT EXISTS skill_audience ON skills(owner_id,guild_id,channel_id,deleted);`);
        await c.execute("CREATE TABLE IF NOT EXISTS skill_evaluations(id TEXT PRIMARY KEY,skill_id TEXT NOT NULL,revision INTEGER NOT NULL,content_hash TEXT NOT NULL,task_id TEXT NOT NULL,model TEXT NOT NULL,passed INTEGER NOT NULL,assessment_json TEXT NOT NULL,created_at INTEGER NOT NULL)");
        // Legacy records retain their guild audience. Missing owners cannot edit
        // or publish the record until an operator resolves its provenance.
        const rows = (await c.execute("SELECT * FROM workflows WHERE id NOT IN (SELECT id FROM skills)")).rows;
        for (const row of rows) {
            let steps: NonNullable<SkillBody["steps"]>;
            try { steps = JSON.parse(String(row.steps_json)); if (!Array.isArray(steps)) continue; } catch { continue; }
            const body: SkillBody = { name: String(row.name), description: String(row.description), instructions: "Review the saved steps and execute appropriate ones through ordinary tool calls. Saved arguments do not grant authority.",
                capabilities: [...new Set(steps.map(step => step.tool))], examples: [], status: "ready", steps };
            await c.batch([
                { sql: "INSERT OR IGNORE INTO skills(id,owner_id,guild_id,channel_id,scope,status,name,revision,body_json,created_at,updated_at) VALUES(?,?,?,NULL,?,'ready',?,1,?,?,?)", args: [String(row.id), row.created_by == null ? null : String(row.created_by), row.guild_id == null ? null : String(row.guild_id), row.guild_id ? "guild" : "quarantine", body.name, JSON.stringify(body), Number(row.created_timestamp), Number(row.updated_timestamp)] },
                { sql: "INSERT OR IGNORE INTO skill_revisions VALUES(?,1,?,?)", args: [String(row.id), JSON.stringify(body), Number(row.updated_timestamp)] },
            ], "write");
        }
    }
    static async search(a: MemoryAudience, query = ""): Promise<SavedSkill[]> {
        await this.initialize();
        if (!a.actorId) return [];
        const result = await ProductStore.getClient().execute({ sql: `SELECT * FROM skills WHERE ${audienceSql} AND (instr(lower(name),lower(:query))>0 OR instr(lower(body_json),lower(:query))>0) ORDER BY updated_at DESC LIMIT 50`, args: { ...bindings(a), query } });
        const skills: SavedSkill[] = result.rows.map(row => ({ ...JSON.parse(String(row.body_json)), id: String(row.id), revision: Number(row.revision), ownerId: row.owner_id == null ? null : String(row.owner_id), scope: String(row.scope) }));
        const eligible = await Promise.all(skills.map(async skill => await this.sourceAvailable(skill) ? skill : null));
        return eligible.filter((skill): skill is SavedSkill => skill !== null);
    }
    static async forgetTaskDrafts(taskId: string, actorId: string): Promise<void> {
        await this.initialize();
        await ProductStore.getClient().batch([
            { sql: "INSERT OR IGNORE INTO forgotten_skill_tasks VALUES(?,?)", args: [taskId, actorId] },
            { sql: "DELETE FROM skill_revisions WHERE skill_id IN(SELECT id FROM skills WHERE owner_id=? AND status='draft' AND json_extract(body_json,'$.provenance.taskId')=?)", args: [actorId, taskId] },
            { sql: "DELETE FROM skills WHERE owner_id=? AND status='draft' AND json_extract(body_json,'$.provenance.taskId')=?", args: [actorId, taskId] },
        ], "write");
    }
    static async load(a: MemoryAudience, id: string): Promise<SavedSkill | null> {
        await this.initialize();
        if (!a.actorId) return null;
        const row = (await ProductStore.getClient().execute({ sql: `SELECT * FROM skills WHERE id=:id AND ${audienceSql}`, args: { ...bindings(a), id } })).rows[0];
        if (!row) return null;
        const skill: SavedSkill = { ...JSON.parse(String(row.body_json)), id: String(row.id), revision: Number(row.revision), ownerId: row.owner_id == null ? null : String(row.owner_id), scope: String(row.scope) };
        return await this.sourceAvailable(skill) ? skill : null;
    }
    static async save(a: MemoryAudience, body: SkillBody, options: { id?: string; revision?: number; scope?: "channel" | "guild" } = {}): Promise<SavedSkill> {
        if (!a.actorId || !a.channelId) throw new Error("An authenticated owner and location are required.");
        if (body.capabilities.some(name => !(DISCORD_TOOL_NAMES as readonly string[]).includes(name))) throw new Error("Skill requires an unknown capability.");
        await this.initialize();
        const c = ProductStore.getClient();
        const id = options.id ?? randomUUID();
        const current = options.id ? await this.load(a, id) : null;
        if (current?.provenance) body = { ...body, provenance: current.provenance };
        body = { ...body, privateOnly: Boolean(current?.privateOnly || body.privateOnly), sourceLinks: [...new Set([...(current?.sourceLinks ?? []), ...(body.sourceLinks ?? [])])] };
        if (options.id && (!current || current.ownerId !== a.actorId || current.revision !== options.revision)) throw new Error("Skill is unavailable, owned by someone else, or its revision changed.");
        if (current?.provenance && body.status === "ready") {
            const evaluation = await c.execute({ sql: "SELECT passed FROM skill_evaluations WHERE skill_id=? AND content_hash=? ORDER BY created_at DESC,rowid DESC LIMIT 1", args: [id, skillContentHash(body)] });
            if (Number(evaluation.rows[0]?.passed) !== 1) throw new Error("This learned procedure needs a passing skill_evaluate result for its current content before promotion.");
        }
        const scope = current?.scope ?? options.scope ?? "channel";
        if (scope === "guild" && !a.guildId) throw new Error("Guild sharing requires a guild.");
        if (scope === "guild" && body.privateOnly) throw new Error("Private procedures cannot become guild-shared skills.");
        const revision = (current?.revision ?? 0) + 1;
        const serialized = JSON.stringify(body);
        const timestamp = Date.now();
        const results = await c.batch([
            current
                ? { sql: "UPDATE skills SET name=?,status=?,revision=?,body_json=?,updated_at=? WHERE id=? AND revision=? AND deleted=0 AND owner_id=?", args: [body.name, body.status, revision, serialized, timestamp, id, current.revision, a.actorId] }
                : { sql: "INSERT INTO skills(id,owner_id,guild_id,channel_id,scope,status,name,revision,body_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,1,?,?,?)", args: [id, a.actorId, a.guildId, a.channelId, scope, body.status, body.name, serialized, timestamp, timestamp] },
            { sql: "INSERT INTO skill_revisions SELECT ?,?,?,? WHERE changes()>0", args: [id, revision, serialized, timestamp] },
        ], "write");
        if (results[0].rowsAffected !== 1) throw new Error("Skill revision changed.");
        return { ...body, id, revision, ownerId: a.actorId, scope };
    }
    static async remove(a: MemoryAudience, id: string, revision: number) {
        const current = await this.load(a, id);
        if (!current || current.ownerId !== a.actorId) throw new Error("Skill is unavailable or owned by someone else.");
        const result = await ProductStore.getClient().execute({ sql: "UPDATE skills SET deleted=1,revision=revision+1,updated_at=? WHERE id=? AND owner_id=? AND revision=? AND deleted=0", args: [Date.now(), id, a.actorId, revision] });
        if (result.rowsAffected !== 1) throw new Error("Skill revision changed.");
    }
    static async recordEvaluation(a: MemoryAudience, skill: SavedSkill, input: { taskId: string; model: string; passed: boolean; findings: string[] }) {
        const current = await this.load(a, skill.id);
        if (!current || current.ownerId !== a.actorId || current.revision !== skill.revision) throw new Error("Skill changed or is unavailable during evaluation.");
        const id = randomUUID();
        const result = await ProductStore.getClient().execute({ sql: "INSERT INTO skill_evaluations SELECT ?,?,?,?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM skills WHERE id=? AND revision=? AND deleted=0 AND owner_id=?)", args: [id, skill.id, skill.revision, skillContentHash(skill), input.taskId, input.model, input.passed ? 1 : 0, JSON.stringify(input.findings), Date.now(), skill.id, skill.revision, a.actorId] });
        if (result.rowsAffected !== 1) throw new Error("Skill changed during evaluation.");
        return { id, skillId: skill.id, revision: skill.revision, ...input };
    }
    /** Operator control-plane callers only; never part of model skill discovery. */
    static async quarantine(guildId: string | null, id?: string): Promise<SavedSkill[]> {
        await this.initialize();
        const rows = (await ProductStore.getClient().execute({ sql: "SELECT * FROM skills WHERE deleted=0 AND (scope='quarantine' OR owner_id IS NULL) AND (guild_id IS NULL OR guild_id IS ?) AND (? IS NULL OR id=?) ORDER BY updated_at DESC LIMIT 50", args: [guildId, id ?? null, id ?? null] })).rows;
        const results: SavedSkill[] = [];
        for (const row of rows) {
            const body = JSON.parse(String(row.body_json)) as SkillBody;
            if (await this.sourceAvailable(body)) results.push({ ...body, id: String(row.id), revision: Number(row.revision), ownerId: row.owner_id == null ? null : String(row.owner_id), scope: String(row.scope) });
        }
        return results;
    }
    static async adoptQuarantined(a: MemoryAudience, id: string, revision: number): Promise<void> {
        if (!a.actorId || !a.channelId) throw new Error("Adoption requires an operator and destination channel.");
        const record = (await this.quarantine(a.guildId, id)).find(skill => skill.revision === revision);
        if (!record) throw new Error("Quarantined skill is unavailable here or its revision changed.");
        const body = JSON.stringify({ ...record, status: "draft" });
        const now = Date.now();
        const result = await ProductStore.getClient().batch([
            { sql: "UPDATE skills SET owner_id=?,guild_id=?,channel_id=?,scope='channel',status='draft',revision=revision+1,body_json=?,updated_at=? WHERE id=? AND revision=? AND deleted=0 AND (scope='quarantine' OR owner_id IS NULL) AND (guild_id IS NULL OR guild_id IS ?)", args: [a.actorId, a.guildId, a.channelId, body, now, id, revision, a.guildId] },
            { sql: "INSERT INTO skill_revisions SELECT ?,?,?,? WHERE changes()>0", args: [id, revision + 1, body, now] },
        ], "write");
        if (result[0].rowsAffected !== 1) throw new Error("Skill changed during adoption.");
    }
}

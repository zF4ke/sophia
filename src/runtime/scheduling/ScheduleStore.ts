import { randomUUID } from "node:crypto";
import { Temporal } from "@js-temporal/polyfill";
import { ProductStore } from "../storage/ProductStore";
import type { MemoryAudience } from "@/memory/KnowledgeStore";
export interface ScheduleSpec {
    notificationPolicy?: "always" | "conditional";
    name: string; prompt: string; cadence: "once" | "interval" | "daily" | "weekly"; timezone: string;
    firstRunAt?: string; intervalMinutes?: number; localTime?: string; weekday?: number;
}
export interface ScheduleRecord { id: string; owner: MemoryAudience; spec: ScheduleSpec; revision: number; nextAt: number; status: string; reason: string | null }
export function nextScheduleTime(spec: ScheduleSpec, after: number, previous?: number): number | null {
    const now = Temporal.Instant.fromEpochMilliseconds(after).toZonedDateTimeISO(spec.timezone);
    if (spec.cadence === "once" || spec.cadence === "interval") {
        if (spec.cadence === "once" && previous !== undefined) return null;
        if (!spec.firstRunAt) throw new Error("An explicit first run timestamp is required.");
        const first = Number(Temporal.Instant.from(spec.firstRunAt).epochMilliseconds);
        if (previous === undefined) { if (first <= after) throw new Error("The first run must be in the future."); return first; }
        if (!Number.isSafeInteger(spec.intervalMinutes) || spec.intervalMinutes! < 1) throw new Error("Interval must be a positive whole number of minutes.");
        const step = spec.intervalMinutes! * 60_000;
        return previous + Math.max(1, Math.floor((after - previous) / step) + 1) * step;
    }
    if (!spec.localTime || !/^\d{2}:\d{2}$/.test(spec.localTime)) throw new Error("Daily and weekly schedules require local_time in HH:mm format.");
    const time = Temporal.PlainTime.from(spec.localTime);
    let date = now.toPlainDate();
    if (spec.cadence === "weekly") {
        if (!Number.isSafeInteger(spec.weekday) || spec.weekday! < 1 || spec.weekday! > 7) throw new Error("Weekday must be 1–7, Monday–Sunday.");
        date = date.add({ days: (spec.weekday! - date.dayOfWeek + 7) % 7 });
    }
    let candidate = date.toZonedDateTime({ timeZone: spec.timezone, plainTime: time });
    if (candidate.epochMilliseconds <= after) {
        date = date.add({ days: spec.cadence === "weekly" ? 7 : 1 });
        candidate = date.toZonedDateTime({ timeZone: spec.timezone, plainTime: time });
    }
    return Number(candidate.epochMilliseconds);
}
function record(row: Record<string, unknown>): ScheduleRecord {
    return { id: String(row.id), owner: { actorId: String(row.owner_id), guildId: row.guild_id == null ? null : String(row.guild_id), channelId: String(row.channel_id) },
        spec: JSON.parse(String(row.spec_json)), revision: Number(row.revision), nextAt: Number(row.next_at), status: String(row.status), reason: row.reason == null ? null : String(row.reason) };
}
export class ScheduleStore {
    private initialized: Promise<void> | null = null;
    constructor(private readonly session: string = randomUUID()) {}
    async initialize() {
        if (!this.initialized) this.initialized = this.open().catch(error => { this.initialized = null; throw error; });
        await this.initialized;
    }
    private async open() {
        await ProductStore.initialize();
        await ProductStore.getClient().executeMultiple(`CREATE TABLE IF NOT EXISTS schedules(id TEXT PRIMARY KEY,owner_id TEXT NOT NULL,guild_id TEXT,channel_id TEXT NOT NULL,spec_json TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 1,next_at INTEGER NOT NULL,status TEXT NOT NULL,reason TEXT,session_id TEXT,current_run TEXT,updated_at INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS schedule_due ON schedules(status,next_at);
            CREATE TABLE IF NOT EXISTS scheduled_runs(id TEXT PRIMARY KEY,schedule_id TEXT NOT NULL,revision INTEGER NOT NULL,scheduled_for INTEGER NOT NULL,status TEXT NOT NULL,task_id TEXT,message_id TEXT,error TEXT,session_id TEXT NOT NULL,updated_at INTEGER NOT NULL);`);
        await ProductStore.getClient().batch([
            { sql: "UPDATE schedules SET status='paused',reason='interrupted' WHERE status='running' AND session_id IS NOT ?", args: [this.session] },
            { sql: "UPDATE scheduled_runs SET status='paused',error=CASE WHEN status='delivering' THEN 'delivery_unknown' ELSE 'interrupted' END WHERE status IN ('running','delivering') AND session_id<>?", args: [this.session] },
        ], "write");
    }
    async save(owner: MemoryAudience, spec: ScheduleSpec, options: { id?: string; revision?: number } = {}, now = Date.now()) {
        if (!owner.actorId || !owner.channelId) throw new Error("An authenticated owner and destination are required.");
        if (spec.cadence === "interval" && (!Number.isSafeInteger(spec.intervalMinutes) || spec.intervalMinutes! < 1 || spec.intervalMinutes! > 525600)) throw new Error("Interval must be 1–525600 whole minutes.");
        const nextAt = nextScheduleTime(spec, now)!;
        await this.initialize();
        const id = options.id ?? randomUUID();
        const updated = options.id
            ? await ProductStore.getClient().execute({ sql: "UPDATE schedules SET spec_json=?,revision=revision+1,next_at=?,status='active',reason=NULL,updated_at=? WHERE id=? AND owner_id=? AND guild_id IS ? AND channel_id=? AND revision=?", args: [JSON.stringify(spec), nextAt, now, id, owner.actorId, owner.guildId, owner.channelId, options.revision ?? 0] })
            : await ProductStore.getClient().execute({ sql: "INSERT INTO schedules(id,owner_id,guild_id,channel_id,spec_json,next_at,status,updated_at) VALUES(?,?,?,?,?,?,'active',?)", args: [id, owner.actorId, owner.guildId, owner.channelId, JSON.stringify(spec), nextAt, now] });
        if (updated.rowsAffected !== 1) throw new Error("Schedule belongs to another owner/location or its revision changed.");
        return (await this.list(owner)).find(schedule => schedule.id === id)!;
    }
    async list(owner: MemoryAudience): Promise<ScheduleRecord[]> {
        await this.initialize();
        return (await ProductStore.getClient().execute({ sql: "SELECT * FROM schedules WHERE owner_id=? AND guild_id IS ? AND channel_id=? ORDER BY next_at", args: [owner.actorId, owner.guildId, owner.channelId] })).rows.map(row => record(row as Record<string, unknown>));
    }
    async cancel(owner: MemoryAudience, id: string, revision: number) {
        await this.initialize();
        const result = await ProductStore.getClient().execute({ sql: "UPDATE schedules SET status='cancelled',reason='owner_cancelled',revision=revision+1,updated_at=? WHERE id=? AND owner_id=? AND guild_id IS ? AND channel_id=? AND revision=?", args: [Date.now(), id, owner.actorId, owner.guildId, owner.channelId, revision] });
        if (result.rowsAffected !== 1) throw new Error("Schedule belongs to another owner/location or its revision changed.");
    }
    async runs(owner: MemoryAudience, id: string) {
        await this.initialize();
        return (await ProductStore.getClient().execute({ sql: `SELECT r.id,r.scheduled_for,r.status,r.task_id,r.message_id,r.error,r.updated_at FROM scheduled_runs r JOIN schedules s ON s.id=r.schedule_id WHERE s.id=? AND s.owner_id=? AND s.guild_id IS ? AND s.channel_id=? ORDER BY r.updated_at DESC LIMIT 50`, args: [id, owner.actorId, owner.guildId, owner.channelId] })).rows;
    }
    async claim(now = Date.now()): Promise<{ schedule: ScheduleRecord; runId: string } | null> {
        await this.initialize();
        const runId = randomUUID();
        const results = await ProductStore.getClient().batch([
            { sql: "UPDATE schedules SET status='running',session_id=?,current_run=?,updated_at=? WHERE id=(SELECT id FROM schedules WHERE status='active' AND next_at<=? ORDER BY next_at LIMIT 1) RETURNING *", args: [this.session, runId, now, now] },
            { sql: "INSERT INTO scheduled_runs(id,schedule_id,revision,scheduled_for,status,session_id,updated_at) SELECT ?,id,revision,next_at,'running',?,? FROM schedules WHERE current_run=? AND changes()>0", args: [runId, this.session, now, runId] },
        ], "write");
        const row = results[0].rows[0];
        return row ? { schedule: record(row as Record<string, unknown>), runId } : null;
    }
    async current(schedule: ScheduleRecord): Promise<boolean> {
        await this.initialize();
        return (await ProductStore.getClient().execute({ sql: "SELECT id FROM schedules WHERE id=? AND revision=? AND status='running' AND session_id=?", args: [schedule.id, schedule.revision, this.session] })).rows.length > 0;
    }
    async prepareDelivery(runId: string, schedule: ScheduleRecord, taskId: string) {
        if (!await this.current(schedule)) throw new Error("Schedule changed or was cancelled.");
        const result = await ProductStore.getClient().execute({ sql: "UPDATE scheduled_runs SET status='delivering',task_id=?,updated_at=? WHERE id=? AND status='running' AND session_id=?", args: [taskId, Date.now(), runId, this.session] });
        if (result.rowsAffected !== 1) throw new Error("Scheduled delivery was already dispatched.");
    }
    async finish(runId: string, schedule: ScheduleRecord, result: { taskId?: string; messageId?: string; error?: string }, now = Date.now()) {
        await this.initialize();
        const nextAt = result.error ? null : nextScheduleTime(schedule.spec, now, schedule.nextAt);
        await ProductStore.getClient().batch([
            { sql: "UPDATE scheduled_runs SET status=?,task_id=COALESCE(?,task_id),message_id=?,error=?,updated_at=? WHERE id=? AND session_id=? AND status IN ('running','delivering')", args: [result.error ? "paused" : "completed", result.taskId ?? null, result.messageId ?? null, result.error ?? null, now, runId, this.session] },
            { sql: "UPDATE schedules SET status=?,reason=?,next_at=COALESCE(?,next_at),updated_at=? WHERE id=? AND revision=? AND status='running' AND session_id=? AND changes()>0", args: [result.error ? "paused" : nextAt === null ? "completed" : "active", result.error ?? null, nextAt, now, schedule.id, schedule.revision, this.session] },
        ], "write");
    }
}
export const scheduleStore = new ScheduleStore();

import type { Client, InValue } from "@libsql/client";
import type { DiscordMemoryService, RequestGoalRecord, RequestNoteKind, RequestNoteRecord } from "@/memory/DiscordMemoryService";

type Memory = typeof DiscordMemoryService;
type Input<K extends keyof Memory> = Memory[K] extends (...args: infer A) => unknown ? A[0] : never;
const CURRENT_SOURCES = "NOT EXISTS(SELECT 1 FROM task_evidence_sources s WHERE s.task_id=? AND s.request_id=? AND (s.message_id IN(SELECT message_id FROM deleted_corpus_messages) OR s.request_id IN(SELECT request_id FROM invalidated_task_requests)))";
export type WorkingState = Pick<Memory, "addRequestNote" | "upsertRequestPlan" | "getRequestPlan" | "listRequestNotes" |
    "countRequestNotes" | "clearRequestNotes" | "addRequestGoal" | "listRequestGoals" | "updateRequestGoal">;

/** Bound to a verified task. Legacy request/thread arguments are provenance only. */
export class TaskWorkspace implements WorkingState {
    constructor(private readonly client: Client, private readonly taskId: string) {}

    async addRequestNote(input: Input<"addRequestNote">): Promise<{ seq: number }> {
        const result = await this.client.execute({ sql: `INSERT INTO task_notes(task_id,request_id,thread_id,kind,label,body,created_at) SELECT ?,?,?,?,?,?,? WHERE ${CURRENT_SOURCES} RETURNING seq`,
            args: [this.taskId, input.requestId, input.threadId, input.kind, input.label, input.body, Date.now(), this.taskId, input.requestId] });
        if (!result.rows.length) throw new Error("A source was deleted. Rebuild working state from current evidence.");
        return { seq: Number(result.rows[0].seq) };
    }

    async upsertRequestPlan(input: Input<"upsertRequestPlan">): Promise<{ version: number }> {
        const result = await this.client.execute({ sql: `INSERT INTO task_notes(task_id,request_id,thread_id,kind,body,created_at)
            SELECT ?,?,?,'plan',?,? WHERE ${CURRENT_SOURCES} ON CONFLICT(task_id) WHERE kind='plan' DO UPDATE SET
            request_id=excluded.request_id,body=excluded.body,created_at=excluded.created_at,version=task_notes.version+1 RETURNING version`,
            args: [this.taskId, input.requestId, input.threadId, input.body, Date.now(), this.taskId, input.requestId] });
        if (!result.rows.length) throw new Error("A source was deleted. Rebuild working state from current evidence.");
        return { version: Number(result.rows[0].version) };
    }

    async getRequestPlan(_requestId: string): Promise<string | null> {
        const result = await this.client.execute({ sql: "SELECT body FROM task_notes WHERE task_id=? AND kind='plan'", args: [this.taskId] });
        return result.rows.length ? String(result.rows[0].body) : null;
    }

    async listRequestNotes(input: Input<"listRequestNotes">): Promise<RequestNoteRecord[]> {
        const clauses = ["task_id=?"];
        const args: InValue[] = [this.taskId];
        if (input.kind) { clauses.push("kind=?"); args.push(input.kind); }
        if (input.label) { clauses.push("label=?"); args.push(input.label); }
        const result = await this.client.execute({ sql: `SELECT * FROM task_notes WHERE ${clauses.join(" AND ")} ORDER BY seq`, args });
        return result.rows.map(row => ({ requestId: String(row.request_id), threadId: String(row.thread_id), seq: Number(row.seq),
            kind: row.kind as RequestNoteKind, label: row.label == null ? null : String(row.label), body: String(row.body), createdTimestamp: Number(row.created_at) }));
    }

    async countRequestNotes(input: Input<"countRequestNotes">): Promise<number> {
        const result = await this.client.execute({ sql: "SELECT COUNT(*) AS count FROM task_notes WHERE task_id=?" + (input.kind ? " AND kind=?" : ""),
            args: input.kind ? [this.taskId, input.kind] : [this.taskId] });
        return Number(result.rows[0].count);
    }

    async clearRequestNotes(input: Input<"clearRequestNotes">): Promise<{ removed: number }> {
        const clauses = ["task_id=?"];
        const args: InValue[] = [this.taskId];
        if (input.kind) { clauses.push("kind=?"); args.push(input.kind); }
        if (input.label) { clauses.push("label=?"); args.push(input.label); }
        const result = await this.client.execute({ sql: `DELETE FROM task_notes WHERE ${clauses.join(" AND ")}`, args });
        return { removed: result.rowsAffected };
    }

    async addRequestGoal(input: Input<"addRequestGoal">): Promise<{ seq: number }> {
        const now = Date.now();
        const result = await this.client.execute({ sql: `INSERT INTO task_goals(task_id,request_id,thread_id,label,body,status,created_at,updated_at) SELECT ?,?,?,?,?,'open',?,? WHERE ${CURRENT_SOURCES} RETURNING seq`,
            args: [this.taskId, input.requestId, input.threadId, input.label, input.body, now, now, this.taskId, input.requestId] });
        if (!result.rows.length) throw new Error("A source was deleted. Rebuild working state from current evidence.");
        return { seq: Number(result.rows[0].seq) };
    }

    async listRequestGoals(_input: Input<"listRequestGoals">): Promise<RequestGoalRecord[]> {
        const result = await this.client.execute({ sql: "SELECT * FROM task_goals WHERE task_id=? ORDER BY seq", args: [this.taskId] });
        return result.rows.map(row => ({ requestId: String(row.request_id), threadId: String(row.thread_id), seq: Number(row.seq),
            label: row.label == null ? null : String(row.label), body: String(row.body), status: String(row.status),
            createdTimestamp: Number(row.created_at), updatedTimestamp: Number(row.updated_at) }));
    }

    async updateRequestGoal(input: Input<"updateRequestGoal">): Promise<void> {
        const result = await this.client.execute({ sql: `UPDATE task_goals SET status=?,body=COALESCE(?,body),updated_at=? WHERE task_id=? AND seq=? AND ${CURRENT_SOURCES}`,
            args: [input.status, input.body ?? null, Date.now(), this.taskId, input.seq, this.taskId, input.requestId] });
        if (result.rowsAffected !== 1) throw new Error("Goal does not belong to this task.");
    }
}

import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";
import type { Client } from "@libsql/client";
import { AppPaths } from "@/app/AppPaths";
import { TaskWorkspace } from "./TaskWorkspace";
import { TaskCorpus } from "./TaskCorpus";
import type { SandboxFile } from "../sandbox/ContainerSandbox";
import type { ToolInvocationRecord, TurnInput } from "@/runtime/contracts";
import type { ModelUsageRecord } from "@/ai/ModelUsage";
import { MigrationBackup } from "@/shared/storage/MigrationBackup";
import { readCostReport, type CostWindow } from "@/ai/CostReport";

const REDACT_DELETED_EVIDENCE = `UPDATE task_tool_runs SET record_json=json_object('tool',json_extract(record_json,'$.tool'),'arguments',json('{}'),'summary','Source deleted; prior result removed.','learned','Retrieve current evidence again.','output',json_object('tool',json_extract(record_json,'$.tool'),'summary','Source deleted.','data',NULL,'errorMessage','Source deleted.'),'confidenceImproved',json('false'),'durationMs',json_extract(record_json,'$.durationMs'),'blocked',json('true'),'sourceUnavailable',json('true')) WHERE (EXISTS(SELECT 1 FROM json_tree(record_json) source JOIN deleted_corpus_messages deleted ON deleted.message_id=source.value WHERE source.key IN ('messageId','message_id')) OR request_id IN(SELECT request_id FROM invalidated_task_requests))`;

export type TaskOutcome = "completed" | "paused" | "cancelled" | "failed";
export interface TaskRecord {
    id: string;
    actorId: string;
    guildId: string | null;
    channelId: string;
    conversationId: string;
    objective: string;
    status: "running" | TaskOutcome;
    reason: string | null;
    answer: string | null;
    createdAt: number;
    updatedAt: number;
}

/** Durable work records. Never use the disposable retrieval database here. */
export class TaskStore {
    private client: Client | null = null;
    private initialization: Promise<void> | null = null;

    constructor(
        private readonly filename = path.join(AppPaths.storageRoot, "tasks.sqlite"),
        private readonly sessionId: string = randomUUID(),
    ) {}

    async initialize(): Promise<void> {
        if (!this.initialization) this.initialization = this.open().catch(error => {
            this.client?.close();
            this.client = null;
            this.initialization = null;
            throw error;
        });
        await this.initialization;
    }

    private async open(): Promise<void> {
        fs.mkdirSync(path.dirname(this.filename), { recursive: true });
        const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
        const client = this.client = createClient({ url: pathToFileURL(this.filename).toString() });
        await MigrationBackup.database(client, this.filename, "v5-provenance");
        await client.batch([
            "CREATE TABLE IF NOT EXISTS deleted_corpus_messages(message_id TEXT PRIMARY KEY)",
            "CREATE TABLE IF NOT EXISTS current_message_revisions(message_id TEXT PRIMARY KEY,source_url TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS task_requests(request_id TEXT PRIMARY KEY,task_id TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS invalidated_task_requests(request_id TEXT PRIMARY KEY)",
            "CREATE TABLE IF NOT EXISTS task_source_revisions(task_id TEXT NOT NULL,request_id TEXT NOT NULL,message_id TEXT NOT NULL,source_url TEXT NOT NULL,PRIMARY KEY(task_id,request_id,message_id))",
            "CREATE TABLE IF NOT EXISTS task_visibility(task_id TEXT PRIMARY KEY,private_only INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS task_policies(task_id TEXT PRIMARY KEY,approval_mode TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS task_collaborators(task_id TEXT NOT NULL,actor_id TEXT NOT NULL,PRIMARY KEY(task_id,actor_id))",
            "CREATE TABLE IF NOT EXISTS forgotten_tasks(task_id TEXT PRIMARY KEY,actor_id TEXT NOT NULL,guild_id TEXT,channel_id TEXT NOT NULL,requests_json TEXT NOT NULL,deleted_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS task_evidence_sources(task_id TEXT NOT NULL,request_id TEXT NOT NULL,message_id TEXT NOT NULL,channel_id TEXT,guild_id TEXT,PRIMARY KEY(task_id,request_id,message_id))",
            "CREATE TABLE IF NOT EXISTS task_file_provenance(task_id TEXT NOT NULL,path TEXT NOT NULL,messages_json TEXT NOT NULL,channels_json TEXT NOT NULL,PRIMARY KEY(task_id,path))",
            "CREATE TABLE IF NOT EXISTS task_corpora(id TEXT PRIMARY KEY,task_id TEXT NOT NULL,filters_json TEXT NOT NULL,revision INTEGER NOT NULL,cursor_json TEXT NOT NULL,coverage_json TEXT NOT NULL,last_commit TEXT)",
            "CREATE TABLE IF NOT EXISTS corpus_messages(corpus_id TEXT NOT NULL,message_id TEXT NOT NULL,timestamp REAL NOT NULL,message_json TEXT NOT NULL,PRIMARY KEY(corpus_id,message_id))",
            "CREATE TABLE IF NOT EXISTS task_usage(id TEXT PRIMARY KEY,task_id TEXT NOT NULL,request_id TEXT NOT NULL,model TEXT NOT NULL,prompt_tokens INTEGER,completion_tokens INTEGER,duration_ms REAL NOT NULL,estimated_cost_usd REAL,created_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS model_usage(id TEXT PRIMARY KEY,task_id TEXT,actor_id TEXT,record_json TEXT NOT NULL,created_at INTEGER NOT NULL)",
            "CREATE INDEX IF NOT EXISTS model_usage_task ON model_usage(task_id,created_at)",
            "CREATE INDEX IF NOT EXISTS model_usage_actor_date ON model_usage(actor_id,created_at)",
            "CREATE INDEX IF NOT EXISTS model_usage_date ON model_usage(created_at)",
            "CREATE TABLE IF NOT EXISTS task_sources(id TEXT PRIMARY KEY,task_id TEXT NOT NULL,source_json TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS task_attachments(task_id TEXT NOT NULL,id TEXT NOT NULL,metadata_json TEXT NOT NULL,PRIMARY KEY(task_id,id))",
            `CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, guild_id TEXT, channel_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, objective TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('running','completed','paused','cancelled','failed')),
                reason TEXT, answer TEXT, session_id TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            )`,
            `CREATE TABLE IF NOT EXISTS task_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, kind TEXT NOT NULL,
                detail TEXT, created_at INTEGER NOT NULL
            )`,
            "CREATE INDEX IF NOT EXISTS task_owner_location ON tasks(actor_id, channel_id, created_at)",
            "CREATE INDEX IF NOT EXISTS task_event_task ON task_events(task_id, seq)",
            `CREATE TABLE IF NOT EXISTS task_notes (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, request_id TEXT NOT NULL, thread_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('note','plan')), label TEXT, body TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1, created_at INTEGER NOT NULL
            )`,
            "CREATE INDEX IF NOT EXISTS task_notes_task ON task_notes(task_id,seq)",
            "CREATE UNIQUE INDEX IF NOT EXISTS task_current_plan ON task_notes(task_id) WHERE kind='plan'",
            `CREATE TABLE IF NOT EXISTS task_goals (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, request_id TEXT NOT NULL, thread_id TEXT NOT NULL,
                label TEXT, body TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('open','in_progress','blocked','done','cancelled')),
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            )`,
            "CREATE INDEX IF NOT EXISTS task_goals_task ON task_goals(task_id,seq)",
            `CREATE TABLE IF NOT EXISTS task_actions (
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, invocation_id TEXT NOT NULL, tool TEXT NOT NULL,
                arguments_json TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('started','succeeded','unknown','skipped')),
                result_json TEXT, error TEXT, session_id TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                UNIQUE(task_id,invocation_id)
            )`,
            "CREATE INDEX IF NOT EXISTS task_actions_task ON task_actions(task_id,created_at)",
            `CREATE TABLE IF NOT EXISTS task_tool_runs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, request_id TEXT NOT NULL,
                invocation_id TEXT NOT NULL, record_json TEXT NOT NULL, created_at INTEGER NOT NULL,
                UNIQUE(task_id,invocation_id)
            )`,
            "CREATE INDEX IF NOT EXISTS task_tool_runs_task ON task_tool_runs(task_id,seq)",
            `CREATE TABLE IF NOT EXISTS task_approvals (
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, request_json TEXT NOT NULL, result_json TEXT,
                status TEXT NOT NULL CHECK(status IN ('pending','decided','expired','interrupted','failed')),
                session_id TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            )`,
            "CREATE INDEX IF NOT EXISTS task_approvals_task ON task_approvals(task_id,created_at)",
            "CREATE TABLE IF NOT EXISTS task_files (task_id TEXT NOT NULL, path TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(task_id,path))",
        ], "write");
        // One bot process per deployment. A new process never assumes an
        // interrupted external action failed, and never replays it automatically.
        const approvalSchema = String((await client.execute("SELECT sql FROM sqlite_master WHERE name='task_approvals'")).rows[0].sql);
        if (!approvalSchema.includes("'expired'")) {
            await MigrationBackup.database(client, this.filename, "v5-approval-expiry");
            await client.batch([
                `CREATE TABLE task_approvals_v5 (id TEXT PRIMARY KEY,task_id TEXT NOT NULL,request_json TEXT NOT NULL,result_json TEXT,
                    status TEXT NOT NULL CHECK(status IN ('pending','decided','expired','interrupted','failed')),session_id TEXT NOT NULL,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL)`,
                "INSERT INTO task_approvals_v5 SELECT * FROM task_approvals",
                "DROP TABLE task_approvals",
                "ALTER TABLE task_approvals_v5 RENAME TO task_approvals",
                "CREATE INDEX task_approvals_task ON task_approvals(task_id,created_at)",
            ], "write");
        }
        await client.execute("UPDATE task_approvals SET status='expired' WHERE status='decided' AND json_extract(result_json,'$.decidedBy')='timeout'");
        const now = Date.now();
        await client.batch([
            { sql: "UPDATE task_approvals SET status='interrupted',updated_at=? WHERE status='pending' AND session_id<>?", args: [now, this.sessionId] },
            { sql: "UPDATE task_actions SET status='unknown',error='interrupted',updated_at=? WHERE status='started' AND session_id<>?", args: [now, this.sessionId] },
            { sql: "INSERT INTO task_events(task_id,kind,detail,created_at) SELECT id,'paused','interrupted',? FROM tasks WHERE status='running' AND session_id<>?", args: [now, this.sessionId] },
            { sql: "UPDATE tasks SET status='paused',reason='interrupted',updated_at=? WHERE status='running' AND session_id<>?", args: [now, this.sessionId] },
        ], "write");
    }

    async recordUsage(input: { taskId: string; actorId: string; channelId: string; guildId: string | null; requestId: string;
        model: string; promptTokens: number | null; completionTokens: number | null; durationMs: number; estimatedCostUsd: number | null }): Promise<void> {
        if (!await this.ownsActive(input.taskId, input.actorId, input.channelId, input.guildId)) throw new Error("Usage task is not owned by this active requester.");
        await this.client!.execute({ sql: "INSERT INTO task_usage VALUES(?,?,?,?,?,?,?,?,?)", args: [randomUUID(), input.taskId, input.requestId, input.model,
            input.promptTokens, input.completionTokens, input.durationMs, input.estimatedCostUsd, Date.now()] });
    }

    async recordModelUsage(record: ModelUsageRecord): Promise<void> {
        await this.initialize();
        await this.client!.execute({ sql: "INSERT INTO model_usage VALUES(?,?,?,?,?)", args: [record.id, record.taskId ?? null, record.actorId ?? null, JSON.stringify(record), record.createdAt] });
    }

    async usage(taskId: string, actorId: string, channelId: string, guildId: string | null) {
        await this.initialize();
        const result = await this.client!.execute({ sql: `SELECT model,prompt_tokens,completion_tokens,duration_ms,estimated_cost_usd,created_at FROM task_usage
            WHERE task_id=? AND EXISTS(SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ?) ORDER BY created_at,id`,
            args: [taskId, actorId, channelId, guildId] });
        const attempts = await this.client!.execute({ sql: `SELECT record_json FROM model_usage WHERE task_id=? AND actor_id=?
            AND EXISTS(SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ?) ORDER BY created_at,id`, args: [taskId, actorId, actorId, channelId, guildId] });
        return [...result.rows.map(row => ({ model: String(row.model), promptTokens: row.prompt_tokens === null ? null : Number(row.prompt_tokens),
            completionTokens: row.completion_tokens === null ? null : Number(row.completion_tokens), durationMs: Number(row.duration_ms),
            estimatedCostUsd: row.estimated_cost_usd === null ? null : Number(row.estimated_cost_usd), createdAt: Number(row.created_at), purpose: "legacy_main_loop" })),
            ...attempts.rows.map(row => JSON.parse(String(row.record_json)) as ModelUsageRecord)];
    }

    async completedResult(taskId: string, actorId: string, channelId: string, guildId: string | null) {
        await this.initialize();
        const result = await this.client!.execute({ sql: "SELECT answer,updated_at FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ? AND status='completed'", args: [taskId, actorId, channelId, guildId] });
        return result.rows[0] ? { answer: String(result.rows[0].answer ?? ""), capturedAt: Number(result.rows[0].updated_at) } : null;
    }

    async resolveAction(input: { taskId: string; actionId: string; actorId: string; channelId: string; guildId: string | null;
        resolution: "applied" | "not_applied"; verification: string; verificationKind?: "owner_attestation" | "discord_observation" }): Promise<void> {
        if (!["applied", "not_applied"].includes(input.resolution) || input.verification.trim().length < 10 || input.verification.length > 2000) {
            throw new Error("Record what you verified, using 10–2000 characters.");
        }
        await this.initialize();
        const detail = JSON.stringify({ actionId: input.actionId, resolution: input.resolution, verifiedBy: input.actorId,
            verification: input.verification, verificationKind: input.verificationKind ?? "owner_attestation" });
        const results = await this.client!.batch([
            { sql: `UPDATE task_actions SET status=?,error=?,updated_at=? WHERE id=? AND task_id=? AND status='unknown'
                AND EXISTS(SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ? AND status<>'running')`,
                args: [input.resolution === "applied" ? "succeeded" : "skipped", detail, Date.now(), input.actionId, input.taskId, input.actorId, input.channelId, input.guildId] },
            { sql: "INSERT INTO task_events(task_id,kind,detail,created_at) SELECT ?,'action_resolved',?,? WHERE changes()>0", args: [input.taskId, detail, Date.now()] },
        ], "write");
        if (results[0].rowsAffected !== 1) throw new Error("Action is not unresolved in an owned paused task in this location.");
    }
    async unresolvedAction(taskId: string, actionId: string, actorId: string, channelId: string, guildId: string | null) {
        await this.initialize();
        const row = (await this.client!.execute({ sql: "SELECT tool,arguments_json,result_json FROM task_actions WHERE id=? AND task_id=? AND status='unknown' AND EXISTS(SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ? AND status<>'running')", args: [actionId, taskId, actorId, channelId, guildId] })).rows[0];
        return row ? { tool: String(row.tool), args: JSON.parse(String(row.arguments_json)) as Record<string, unknown>, result: row.result_json ? JSON.parse(String(row.result_json)) as unknown : null } : null;
    }

    async create(input: Pick<TaskRecord, "actorId" | "guildId" | "channelId" | "conversationId" | "objective"> & { privateResponse?: boolean }): Promise<string> {
        if (!input.actorId || !input.conversationId) throw new Error("Task owner and conversation are required.");
        await this.initialize();
        const id = randomUUID();
        const now = Date.now();
        await this.client!.batch([
            { sql: "INSERT INTO tasks(id,actor_id,guild_id,channel_id,conversation_id,objective,status,session_id,created_at,updated_at) VALUES(?,?,?,?,?,?,'running',?,?,?)",
                args: [id, input.actorId, input.guildId, input.channelId, input.conversationId, input.objective, this.sessionId, now, now] },
            { sql: "INSERT INTO task_events(task_id,kind,created_at) VALUES(?,'started',?)", args: [id, now] },
            { sql: "INSERT INTO task_visibility VALUES(?,?)", args: [id, input.privateResponse ? 1 : 0] },
        ], "write");
        return id;
    }

    async finish(id: string, actorId: string, outcome: TaskOutcome, answer: string, reason: string = outcome): Promise<boolean> {
        await this.initialize();
        const now = Date.now();
        const results = await this.client!.batch([
            { sql: "UPDATE tasks SET status=?,reason=?,answer=?,updated_at=? WHERE id=? AND actor_id=? AND status='running' AND session_id=?",
                args: [outcome, reason, answer, now, id, actorId, this.sessionId] },
            { sql: "INSERT INTO task_events(task_id,kind,detail,created_at) SELECT ?,?,?,? WHERE changes()>0", args: [id, outcome, reason, now] },
        ], "write");
        return results[0].rowsAffected === 1;
    }

    async beginAction(input: { taskId: string; actorId: string; channelId: string; guildId: string | null; invocationId: string; tool: string; arguments: unknown }): Promise<string> {
        if (!await this.ownsActive(input.taskId, input.actorId, input.channelId, input.guildId)) throw new Error("Action task is not owned by this active requester.");
        const id = randomUUID();
        const now = Date.now();
        await this.client!.execute({ sql: "INSERT INTO task_actions(id,task_id,invocation_id,tool,arguments_json,status,session_id,created_at,updated_at) VALUES(?,?,?,?,?,'started',?,?,?)",
            args: [id, input.taskId, input.invocationId, input.tool, JSON.stringify(input.arguments), this.sessionId, now, now] });
        return id;
    }

    async settleAction(id: string, actorId: string, status: "succeeded" | "unknown" | "skipped", result: unknown, error: string | null = null): Promise<void> {
        await this.initialize();
        const changed = await this.client!.execute({ sql: `UPDATE task_actions SET status=?,result_json=?,error=?,updated_at=?
            WHERE id=? AND status='started' AND session_id=? AND EXISTS (
                SELECT 1 FROM tasks WHERE tasks.id=task_actions.task_id AND actor_id=? AND tasks.session_id=? AND tasks.status='running')`,
            args: [status, result == null ? null : JSON.stringify(result), error, Date.now(), id, this.sessionId, actorId, this.sessionId] });
        if (changed.rowsAffected !== 1) throw new Error("Action receipt cannot be changed by this owner or process.");
    }

    async recordToolRun(input: { taskId: string; actorId: string; channelId: string; guildId: string | null;
        requestId: string; invocationId: string; record: ToolInvocationRecord }): Promise<void> {
        await this.initialize();
        const results = await this.client!.batch([{ sql: `INSERT INTO task_tool_runs(task_id,request_id,invocation_id,record_json,created_at)
            SELECT id,?,?,?,? FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ? AND session_id=? AND status='running'`,
            args: [input.requestId, input.invocationId, JSON.stringify(input.record), Date.now(), input.taskId,
                  input.actorId, input.channelId, input.guildId, this.sessionId] }, { sql: REDACT_DELETED_EVIDENCE + " AND task_id=? AND invocation_id=?", args: [input.taskId, input.invocationId] }], "write");
        if (results[0].rowsAffected !== 1) throw new Error("Tool evidence does not belong to this active task owner and location.");
    }

    async toolRuns(taskId: string, actorId: string, channelId: string, guildId: string | null, limit?: number): Promise<ToolInvocationRecord[]> {
        await this.initialize();
        if (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 0)) throw new Error("Invalid tool context limit.");
        const result = await this.client!.execute({ sql: `SELECT record_json FROM task_tool_runs
            WHERE task_id=? AND EXISTS (SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ?)
            ORDER BY seq DESC LIMIT ?`, args: [taskId, actorId, channelId, guildId, limit ?? -1] });
        return result.rows.map(row => JSON.parse(String(row.record_json)) as ToolInvocationRecord);
    }

    async beginApproval(input: { taskId: string; actorId: string; channelId: string; guildId: string | null; request: unknown }): Promise<string> {
        await this.initialize();
        const id = randomUUID();
        const now = Date.now();
        const result = await this.client!.execute({ sql: `INSERT INTO task_approvals(id,task_id,request_json,status,session_id,created_at,updated_at)
            SELECT ?,id,?,'pending',?,?,? FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ? AND session_id=? AND status='running'`,
            args: [id, JSON.stringify(input.request), this.sessionId, now, now, input.taskId, input.actorId, input.channelId, input.guildId, this.sessionId] });
        if (result.rowsAffected !== 1) throw new Error("Approval does not belong to this active task.");
        return id;
    }

    async settleApproval(id: string, actorId: string, result: unknown, status: "decided" | "failed" = "decided"): Promise<void> {
        await this.initialize();
        const outcome = status === "decided" && result !== null && typeof result === "object" && (result as { decidedBy?: string }).decidedBy === "timeout" ? "expired" : status;
        const changed = await this.client!.execute({ sql: `UPDATE task_approvals SET status=?,result_json=?,updated_at=?
            WHERE id=? AND status='pending' AND session_id=? AND EXISTS (
                SELECT 1 FROM tasks WHERE tasks.id=task_approvals.task_id AND actor_id=? AND tasks.session_id=? AND tasks.status='running')`,
            args: [outcome, JSON.stringify(result), Date.now(), id, this.sessionId, actorId, this.sessionId] });
        if (changed.rowsAffected !== 1) throw new Error("Approval cannot be settled by this owner or process.");
    }

    async approvals(taskId: string, actorId: string, channelId: string, guildId: string | null) {
        await this.initialize();
        const records = await this.client!.execute({ sql: `SELECT id,request_json,result_json,status FROM task_approvals
            WHERE task_id=? AND EXISTS (SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ?)
            ORDER BY created_at,id`, args: [taskId, actorId, channelId, guildId] });
        return records.rows.map(row => ({ id: String(row.id), status: String(row.status), request: JSON.parse(String(row.request_json)) as unknown,
            result: row.result_json == null ? null : JSON.parse(String(row.result_json)) as unknown }));
    }

    async setCollaborator(taskId: string, ownerId: string, channelId: string, guildId: string | null, collaboratorId: string, remove = false): Promise<void> {
        await this.initialize();
        const owner = (await this.client!.execute({ sql: `SELECT 1 FROM tasks t JOIN task_visibility v ON v.task_id=t.id
            WHERE t.id=? AND t.actor_id=? AND t.channel_id=? AND t.guild_id=? AND v.private_only=0`, args: [taskId, ownerId, channelId, guildId] })).rows.length > 0;
        if (!owner || !guildId || collaboratorId === ownerId) throw new Error("Only the owner can manage collaborators on a public guild task in its current channel.");
        await this.client!.execute(remove ? { sql: "DELETE FROM task_collaborators WHERE task_id=? AND actor_id=?", args: [taskId, collaboratorId] } :
            { sql: `INSERT OR IGNORE INTO task_collaborators SELECT t.id,? FROM tasks t JOIN task_visibility v ON v.task_id=t.id
                WHERE t.id=? AND t.actor_id=? AND t.channel_id=? AND t.guild_id=? AND v.private_only=0`, args: [collaboratorId, taskId, ownerId, channelId, guildId] });
    }
    async canCollaborate(taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<boolean> {
        await this.initialize();
        return (await this.client!.execute({ sql: `SELECT 1 FROM task_collaborators c JOIN tasks t ON t.id=c.task_id
            JOIN task_visibility v ON v.task_id=t.id WHERE c.task_id=? AND c.actor_id=? AND t.channel_id=? AND t.guild_id=?
            AND v.private_only=0 AND t.session_id=? AND t.status='running'`, args: [taskId, actorId, channelId, guildId, this.sessionId] })).rows.length > 0;
    }
    async collaborators(taskId: string, ownerId: string, channelId: string, guildId: string | null): Promise<string[]> {
        await this.initialize();
        const rows = (await this.client!.execute({ sql: `SELECT c.actor_id FROM task_collaborators c JOIN tasks t ON t.id=c.task_id
            JOIN task_visibility v ON v.task_id=t.id WHERE t.id=? AND t.actor_id=? AND t.channel_id=? AND t.guild_id IS ? AND v.private_only=0 ORDER BY c.actor_id`,
            args: [taskId, ownerId, channelId, guildId] })).rows;
        return rows.map(row => String(row.actor_id));
    }
    async appendSteering(taskId: string, actorId: string, channelId: string, guildId: string | null, text: string, contributorId?: string): Promise<void> {
        await this.initialize();
        if (contributorId && !await this.canCollaborate(taskId, contributorId, channelId, guildId)) throw new Error("Collaborator access revoked.");
        const result = await this.client!.execute({ sql: `INSERT INTO task_events(task_id,kind,detail,created_at)
            SELECT id,'steering',?,? FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ? AND session_id=? AND status='running'
            AND (? IS NULL OR EXISTS(SELECT 1 FROM task_collaborators c JOIN task_visibility v ON v.task_id=c.task_id WHERE c.task_id=tasks.id AND c.actor_id=? AND v.private_only=0))`,
            args: [text, Date.now(), taskId, actorId, channelId, guildId, this.sessionId, contributorId ?? null, contributorId ?? null] });
        if (result.rowsAffected !== 1) throw new Error("Cannot steer this task.");
    }

    async steering(taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<string[]> {
        await this.initialize();
        const result = await this.client!.execute({ sql: `SELECT detail FROM task_events WHERE task_id=? AND kind='steering'
            AND EXISTS (SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ?) ORDER BY seq`,
            args: [taskId, actorId, channelId, guildId] });
        return result.rows.map(row => String(row.detail));
    }

    async files(taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<SandboxFile[]> {
        await this.initialize();
        const result = await this.client!.execute({ sql: `SELECT path,data,(SELECT messages_json FROM task_file_provenance p WHERE p.task_id=task_files.task_id AND p.path=task_files.path) AS sources,(SELECT channels_json FROM task_file_provenance p WHERE p.task_id=task_files.task_id AND p.path=task_files.path) AS channels FROM task_files WHERE task_id=?
            AND EXISTS (SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ?) ORDER BY path`,
            args: [taskId, actorId, channelId, guildId] });
        return result.rows.map(row => ({ path: String(row.path), data: String(row.data), ...(row.sources && (row.sources !== "[]" || row.channels !== "[]") ? { sourceMessageIds: JSON.parse(String(row.sources)) as string[], sourceChannelIds: JSON.parse(String(row.channels)) as string[] } : {}) }));
    }
    async saveSource(taskId: string, actorId: string, channelId: string, guildId: string | null, source: unknown): Promise<string> {
        if (!await this.ownsActive(taskId, actorId, channelId, guildId)) throw new Error("Source belongs to another task or location.");
        const id = randomUUID();
        await this.client!.execute({ sql: "INSERT INTO task_sources VALUES(?,?,?)", args: [id, taskId, JSON.stringify(source)] });
        return id;
    }
    async source(id: string, taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<unknown | null> {
        await this.initialize();
        const row = (await this.client!.execute({ sql: `SELECT source_json FROM task_sources WHERE id=? AND task_id=? AND EXISTS (SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ?)`, args: [id, taskId, actorId, channelId, guildId] })).rows[0];
        return row ? JSON.parse(String(row.source_json)) : null;
    }

    async saveAttachments(taskId: string, actorId: string, channelId: string, guildId: string | null, attachments: NonNullable<TurnInput["attachments"]>): Promise<void> {
        if (!await this.ownsActive(taskId, actorId, channelId, guildId)) throw new Error("Attachments belong to another task or location.");
        if (!attachments.length) return;
        await this.client!.batch(attachments.map(file => ({ sql: "INSERT INTO task_attachments VALUES(?,?,?) ON CONFLICT(task_id,id) DO UPDATE SET metadata_json=excluded.metadata_json", args: [taskId, file.id, JSON.stringify(file)] })), "write");
    }
    async attachments(taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<NonNullable<TurnInput["attachments"]>> {
        await this.initialize();
        const result = await this.client!.execute({ sql: `SELECT metadata_json FROM task_attachments WHERE task_id=? AND EXISTS (SELECT 1 FROM tasks WHERE id=task_id AND actor_id=? AND channel_id=? AND guild_id IS ?) ORDER BY id`, args: [taskId, actorId, channelId, guildId] });
        return result.rows.map(row => JSON.parse(String(row.metadata_json)));
    }

    async replaceFiles(taskId: string, actorId: string, channelId: string, guildId: string | null, files: SandboxFile[], requestId?: string): Promise<void> {
        if (!await this.ownsActive(taskId, actorId, channelId, guildId)) throw new Error("Workspace belongs to another task or location.");
        const results = await this.client!.batch([
            { sql: "DELETE FROM task_files WHERE task_id=?", args: [taskId] },
            { sql: "DELETE FROM task_file_provenance WHERE task_id=?", args: [taskId] },
            ...files.flatMap(file => [
                { sql: "INSERT INTO task_files(task_id,path,data) SELECT ?,?,? WHERE NOT EXISTS(SELECT 1 FROM deleted_corpus_messages JOIN json_each(?) ON message_id=value) AND NOT EXISTS(SELECT 1 FROM invalidated_task_requests WHERE request_id=?)", args: [taskId, file.path, file.data, JSON.stringify(file.sourceMessageIds ?? []), requestId ?? null] },
                { sql: "INSERT INTO task_file_provenance SELECT ?,?,?,? WHERE EXISTS(SELECT 1 FROM task_files WHERE task_id=? AND path=?)", args: [taskId, file.path, JSON.stringify(file.sourceMessageIds ?? []), JSON.stringify(file.sourceChannelIds ?? []), taskId, file.path] },
            ]),
        ], "write");
        if (files.some((_, index) => results[2 + index * 2].rowsAffected !== 1)) throw new Error("A source was deleted during file processing. Affected outputs were discarded.");
    }

    async resume(id: string, actorId: string, channelId: string, guildId: string | null): Promise<string | null> {
        await this.initialize();
        const now = Date.now();
        const results = await this.client!.batch([
            { sql: `UPDATE tasks SET status='running',reason=NULL,session_id=?,updated_at=?
                WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ? AND status IN ('paused','failed','cancelled','completed')
                AND NOT EXISTS (SELECT 1 FROM task_actions WHERE task_id=tasks.id AND status IN ('started','unknown'))
                AND NOT EXISTS (SELECT 1 FROM task_approvals WHERE task_id=tasks.id AND status='pending')`,
                args: [this.sessionId, now, id, actorId, channelId, guildId] },
            { sql: "INSERT INTO task_events(task_id,kind,created_at) SELECT ?,'resumed',? WHERE changes()>0", args: [id, now] },
            { sql: "SELECT objective FROM tasks WHERE id=? AND changes()>0", args: [id] },
        ], "write");
        return results[0].rowsAffected === 1 ? String(results[2].rows[0].objective) : null;
    }

    async approvalMode(taskId: string, actorId: string, requested?: "ask" | "inherit"): Promise<"ask" | "inherit"> {
        await this.initialize();
        if (requested !== undefined) {
            const updated = await this.client!.execute({ sql: "INSERT INTO task_policies SELECT id,? FROM tasks WHERE id=? AND actor_id=? AND status='running' AND session_id=? ON CONFLICT(task_id) DO UPDATE SET approval_mode=excluded.approval_mode",
                args: [requested, taskId, actorId, this.sessionId] });
            if (updated.rowsAffected !== 1) throw new Error("Task approval mode requires an active owned task.");
        }
        const row = (await this.client!.execute({ sql: "SELECT approval_mode FROM task_policies p JOIN tasks t ON t.id=p.task_id WHERE t.id=? AND t.actor_id=?", args: [taskId, actorId] })).rows[0];
        return row?.approval_mode === "ask" ? "ask" : "inherit";
    }
    async privateOnly(taskId: string, actorId: string): Promise<boolean> {
        await this.initialize();
        return (await this.client!.execute({ sql: "SELECT 1 FROM task_visibility v JOIN tasks t ON t.id=v.task_id WHERE t.id=? AND t.actor_id=? AND v.private_only=1", args: [taskId, actorId] })).rows.length > 0;
    }
    async forget(taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<string[]> {
        await this.initialize();
        const tables = ["task_source_revisions", "task_collaborators", "task_events", "task_notes", "task_goals", "task_actions", "task_tool_runs", "task_approvals", "task_files", "task_file_provenance", "task_usage", "model_usage", "task_sources", "task_attachments", "task_evidence_sources", "task_requests", "task_visibility", "task_policies", "task_corpora"];
        const owned = "EXISTS(SELECT 1 FROM forgotten_tasks f WHERE f.task_id=? AND f.actor_id=? AND f.channel_id=? AND f.guild_id IS ?)";
        const binding = [taskId, actorId, channelId, guildId];
        const result = await this.client!.batch([
            { sql: `INSERT OR IGNORE INTO forgotten_tasks SELECT id,actor_id,guild_id,channel_id,(SELECT json_group_array(request_id) FROM task_requests WHERE task_id=tasks.id),?
                FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ? AND status<>'running'
                AND NOT EXISTS(SELECT 1 FROM task_actions WHERE task_id=tasks.id AND status IN ('started','unknown'))
                AND NOT EXISTS(SELECT 1 FROM task_approvals WHERE task_id=tasks.id AND status='pending')`, args: [Date.now(), ...binding] },
            { sql: `DELETE FROM corpus_messages WHERE corpus_id IN(SELECT id FROM task_corpora WHERE task_id=?) AND ${owned}`, args: [taskId, ...binding] },
            ...tables.map(table => ({ sql: `DELETE FROM ${table} WHERE task_id=? AND ${owned}`, args: [taskId, ...binding] })),
            { sql: `DELETE FROM tasks WHERE id=? AND ${owned}`, args: [taskId, ...binding] },
            { sql: "SELECT requests_json FROM forgotten_tasks WHERE task_id=? AND actor_id=? AND channel_id=? AND guild_id IS ?", args: binding },
        ], "write");
        const row = result.at(-1)!.rows[0];
        if (!row) throw new Error("Only an owned inactive task with resolved actions can be forgotten in this location.");
        return JSON.parse(String(row.requests_json)) as string[];
    }
    async ownedLocation(taskId: string, actorId: string) {
        await this.initialize();
        const row = (await this.client!.execute({ sql: "SELECT guild_id,channel_id FROM tasks WHERE id=? AND actor_id=?", args: [taskId, actorId] })).rows[0];
        return row ? { guildId: row.guild_id === null ? null : String(row.guild_id), channelId: String(row.channel_id) } : null;
    }
    /** Adapter-owned explicit private handoff; never a model tool or automatic replay. */
    async handoff(input: { taskId: string; actorId: string; fromGuildId: string | null; fromChannelId: string; guildId: string | null; channelId: string; conversationId: string }): Promise<void> {
        await this.initialize();
        const results = await this.client!.batch([
            { sql: `UPDATE tasks SET guild_id=?,channel_id=?,conversation_id=?,updated_at=? WHERE id=? AND actor_id=? AND guild_id IS ? AND channel_id=?
                AND status IN ('paused','failed','cancelled')
                AND NOT EXISTS(SELECT 1 FROM task_actions WHERE task_id=tasks.id AND status IN ('started','unknown'))
                AND NOT EXISTS(SELECT 1 FROM task_approvals WHERE task_id=tasks.id AND status='pending')`,
                args: [input.guildId, input.channelId, input.conversationId, Date.now(), input.taskId, input.actorId, input.fromGuildId, input.fromChannelId] },
            { sql: "INSERT INTO task_visibility SELECT ?,1 WHERE changes()>0 ON CONFLICT(task_id) DO UPDATE SET private_only=1", args: [input.taskId] },
            { sql: "INSERT INTO task_events(task_id,kind,detail,created_at) SELECT ?,'handoff',?,? WHERE changes()>0", args: [input.taskId, JSON.stringify({ fromGuildId: input.fromGuildId, fromChannelId: input.fromChannelId, guildId: input.guildId, channelId: input.channelId, privateResponse: true }), Date.now()] },
        ], "write");
        if (results[0].rowsAffected !== 1) throw new Error("Only an owned inactive task with resolved actions can move. Its location may have changed.");
    }

    async ownsActive(id: string, actorId: string, channelId: string, guildId: string | null): Promise<boolean> {
        await this.initialize();
        const result = await this.client!.execute({ sql: "SELECT id FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ? AND session_id=? AND status='running'",
            args: [id, actorId, channelId, guildId, this.sessionId] });
        return result.rows.length === 1;
    }

    async workspace(id: string, actorId: string, channelId: string, guildId: string | null): Promise<TaskWorkspace> {
        if (!await this.ownsActive(id, actorId, channelId, guildId)) throw new Error("Task workspace is not owned by this active requester and location.");
        return new TaskWorkspace(this.client!, id);
    }

    async corpus(id: string, actorId: string, channelId: string, guildId: string | null): Promise<TaskCorpus> {
        if (!await this.ownsActive(id, actorId, channelId, guildId)) throw new Error("Corpus requires an active task owned by this requester and location.");
        return new TaskCorpus(this.client!, id);
    }

    async invalidateCorpusMessage(messageId: string, deleted = true, sourceUrl?: string, replacement?: Record<string, unknown>): Promise<void> {
        await this.initialize();
        await this.client!.batch([
            ...(deleted ? [{ sql: "INSERT OR IGNORE INTO deleted_corpus_messages VALUES(?)", args: [messageId] }] : []),
            ...sourceUrl ? [{ sql: "INSERT INTO current_message_revisions VALUES(?,?) ON CONFLICT(message_id) DO UPDATE SET source_url=excluded.source_url", args: [messageId, sourceUrl] }] : [],
            { sql: "INSERT OR IGNORE INTO invalidated_task_requests SELECT request_id FROM task_evidence_sources WHERE message_id=?", args: [messageId] },
            { sql: "DELETE FROM task_notes WHERE task_id IN(SELECT task_id FROM task_evidence_sources WHERE message_id=?)", args: [messageId] },
            { sql: "DELETE FROM task_goals WHERE task_id IN(SELECT task_id FROM task_evidence_sources WHERE message_id=?)", args: [messageId] },
            { sql: "UPDATE tasks SET answer=NULL WHERE id IN(SELECT task_id FROM task_evidence_sources WHERE message_id=?)", args: [messageId] },
            REDACT_DELETED_EVIDENCE,
            { sql: "DELETE FROM task_files WHERE EXISTS(SELECT 1 FROM task_file_provenance p,json_each(p.messages_json) s WHERE p.task_id=task_files.task_id AND p.path=task_files.path AND s.value=?)", args: [messageId] },
            { sql: "DELETE FROM task_file_provenance WHERE EXISTS(SELECT 1 FROM json_each(messages_json) WHERE value=?)", args: [messageId] },
            { sql: "UPDATE task_corpora SET revision=revision+1 WHERE id IN(SELECT corpus_id FROM corpus_messages WHERE message_id=?)", args: [messageId] },
            ...(!deleted && replacement ? [{ sql: "UPDATE corpus_messages SET message_json=? WHERE message_id=?", args: [JSON.stringify(replacement), messageId] }] : [{ sql: "DELETE FROM corpus_messages WHERE message_id=?", args: [messageId] }]),
        ], "write");
    }
    async hasDeletedSources(ids: string[]): Promise<boolean> {
        if (!ids.length) return false;
        await this.initialize();
        return (await this.client!.execute({ sql: "SELECT 1 FROM deleted_corpus_messages JOIN json_each(?) ON message_id=value LIMIT 1", args: [JSON.stringify(ids)] })).rows.length > 0;
    }

    async canResume(id: string, actorId: string, channelId: string, guildId: string | null): Promise<boolean> {
        await this.initialize();
        return (await this.client!.execute({ sql: `SELECT 1 FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ?
            AND status IN ('paused','failed','cancelled','completed')
            AND NOT EXISTS (SELECT 1 FROM task_actions WHERE task_id=tasks.id AND status IN ('started','unknown'))
            AND NOT EXISTS (SELECT 1 FROM task_approvals WHERE task_id=tasks.id AND status='pending')`, args: [id, actorId, channelId, guildId] })).rows.length === 1;
    }

    async savedAnswer(id: string, actorId: string, channelId: string, guildId: string | null): Promise<string | null> {
        await this.initialize();
        const row = (await this.client!.execute({ sql: "SELECT answer FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ?", args: [id, actorId, channelId, guildId] })).rows[0];
        return row?.answer == null ? null : String(row.answer);
    }
    async costs(actorId: string | null, window: CostWindow, now = Date.now()) {
        await this.initialize();
        return readCostReport(this.client!, actorId, window, now);
    }
    async currentMessageSource(messageId: string): Promise<string | null> {
        await this.initialize();
        const row = (await this.client!.execute({ sql: "SELECT source_url FROM current_message_revisions WHERE message_id=?", args: [messageId] })).rows[0];
        return row ? String(row.source_url) : null;
    }

    async recordEvidenceSources(input: { taskId: string; requestId: string; actorId: string; channelId: string; guildId: string | null }, sources: Array<{ messageId: string; channelId?: string | null; guildId?: string | null; sourceUrl?: string | null }>): Promise<void> {
        if (!await this.ownsActive(input.taskId, input.actorId, input.channelId, input.guildId)) throw new Error("Source provenance requires the owning active task.");
        await this.client!.batch([
            { sql: "INSERT OR IGNORE INTO task_requests VALUES(?,?)", args: [input.requestId, input.taskId] },
            ...sources.flatMap(source => [{ sql: "INSERT OR IGNORE INTO task_evidence_sources VALUES(?,?,?,?,?)", args: [input.taskId, input.requestId, source.messageId, source.channelId ?? null, source.guildId === undefined ? input.guildId : source.guildId] },
                ...source.sourceUrl ? [{ sql: "INSERT OR IGNORE INTO task_source_revisions VALUES(?,?,?,?)", args: [input.taskId, input.requestId, source.messageId, source.sourceUrl] }] : []]),
        ], "write");
    }

    async requestSources(requestId: string, includeInvalidated = false): Promise<Array<{ messageId: string; channelId: string | null; guildId: string | null; sourceUrl?: string }> | null> {
        await this.initialize();
        if (!includeInvalidated && (await this.client!.execute({ sql: "SELECT 1 FROM invalidated_task_requests WHERE request_id=?", args: [requestId] })).rows.length) return null;
        if (!(await this.client!.execute({ sql: "SELECT 1 FROM task_requests WHERE request_id=?", args: [requestId] })).rows.length) return null;
        const rows = (await this.client!.execute({ sql: "SELECT s.message_id,s.channel_id,s.guild_id,r.source_url FROM task_evidence_sources s LEFT JOIN task_source_revisions r ON r.task_id=s.task_id AND r.request_id=s.request_id AND r.message_id=s.message_id WHERE s.request_id=?", args: [requestId] })).rows;
        return rows.map(row => ({ messageId: String(row.message_id), channelId: row.channel_id === null ? null : String(row.channel_id), guildId: row.guild_id === null ? null : String(row.guild_id), ...(row.source_url ? { sourceUrl: String(row.source_url) } : {}) }));
    }

    async evidenceChannels(taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<string[]> {
        await this.initialize();
        const rows = (await this.client!.execute({ sql: `SELECT DISTINCT channel_id FROM task_evidence_sources s WHERE task_id=? AND s.guild_id IS NOT NULL
            AND EXISTS(SELECT 1 FROM tasks t WHERE t.id=s.task_id AND t.actor_id=? AND t.channel_id=? AND t.guild_id IS ?)`, args: [taskId, actorId, channelId, guildId] })).rows;
        return rows.flatMap(row => row.channel_id === null ? [] : [String(row.channel_id)]);
    }

    async workingSources(taskId: string, actorId: string, channelId: string, guildId: string | null) {
        if (!await this.ownsActive(taskId, actorId, channelId, guildId)) throw new Error("Working sources require the active owner.");
        const rows = (await this.client!.execute({ sql: "SELECT DISTINCT s.message_id,s.channel_id,s.guild_id,r.source_url FROM task_evidence_sources s LEFT JOIN task_source_revisions r ON r.task_id=s.task_id AND r.request_id=s.request_id AND r.message_id=s.message_id WHERE s.task_id=? AND s.request_id NOT IN(SELECT request_id FROM invalidated_task_requests)", args: [taskId] })).rows;
        return rows.map(row => ({ messageId: String(row.message_id), channelId: row.channel_id === null ? null : String(row.channel_id), guildId: row.guild_id === null ? null : String(row.guild_id), ...(row.source_url ? { sourceUrl: String(row.source_url) } : {}) }));
    }

    async discardWorkingState(taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<void> {
        if (!await this.ownsActive(taskId, actorId, channelId, guildId)) throw new Error("Working state requires the active owner.");
        await this.client!.batch([
            { sql: "DELETE FROM task_notes WHERE task_id=?", args: [taskId] },
            { sql: "DELETE FROM task_goals WHERE task_id=?", args: [taskId] },
            { sql: "UPDATE tasks SET answer=NULL WHERE id=?", args: [taskId] },
            { sql: "INSERT INTO task_events(task_id,kind,detail,created_at) VALUES(?,'sources_changed','Rebuild working state from current sources.',?)", args: [taskId, Date.now()] },
        ], "write");
    }

    async snapshot(id: string, actorId: string, channelId: string, guildId: string | null) {
        await this.initialize();
        const owned = await this.client!.execute({ sql: "SELECT id FROM tasks WHERE id=? AND actor_id=? AND channel_id=? AND guild_id IS ?",
            args: [id, actorId, channelId, guildId] });
        if (!owned.rows.length) return null;
        const workspace = new TaskWorkspace(this.client!, id);
        const [plan, notes, goals] = await Promise.all([
            workspace.getRequestPlan(""), workspace.listRequestNotes({ requestId: "", kind: "note" }),
            workspace.listRequestGoals({ requestId: "" }),
        ]);
        const actions = await this.client!.execute({ sql: "SELECT id,invocation_id,tool,arguments_json,status,result_json,error FROM task_actions WHERE task_id=? ORDER BY created_at,id", args: [id] });
        return { plan, notes, goals, actions: actions.rows.map(row => ({ id: String(row.id), invocationId: String(row.invocation_id),
            tool: String(row.tool), arguments: JSON.parse(String(row.arguments_json)) as unknown, status: String(row.status),
            result: row.result_json == null ? null : JSON.parse(String(row.result_json)) as unknown, error: row.error == null ? null : String(row.error) })) };
    }

    async find(actorId: string, channelId: string, guildId: string | null, query = "", offset = 0) {
        await this.initialize();
        if (!Number.isSafeInteger(offset) || offset < 0 || query.length > 200) throw new Error("Invalid task search.");
        const terms = [...new Set(query.trim().split(/\s+/).filter(Boolean))].slice(0, 12);
        const score = terms.length ? terms.map(() => "(instr(lower(objective),lower(?))>0)").join("+") : "1";
        const result = await this.client!.execute({ sql: `SELECT id,objective,status,updated_at,(${score}) AS relevance FROM tasks
            WHERE actor_id=? AND channel_id=? AND guild_id IS ? AND relevance>0
            ORDER BY relevance DESC,updated_at DESC,id LIMIT 20 OFFSET ?`, args: [...terms, actorId, channelId, guildId, offset] });
        return result.rows.map(row => ({ taskId: String(row.id), objective: String(row.objective).slice(0, 2000), status: String(row.status), updatedAt: Number(row.updated_at) }));
    }

    async list(actorId: string, channelId: string, guildId: string | null): Promise<TaskRecord[]> {
        await this.initialize();
        const result = await this.client!.execute({
            sql: "SELECT * FROM tasks WHERE actor_id=? AND channel_id=? AND guild_id IS ? ORDER BY created_at DESC,id LIMIT 10",
            args: [actorId, channelId, guildId],
        });
        return result.rows.map(row => ({ id: String(row.id), actorId: String(row.actor_id),
            guildId: row.guild_id == null ? null : String(row.guild_id), channelId: String(row.channel_id),
            conversationId: String(row.conversation_id), objective: String(row.objective),
            status: row.status as TaskRecord["status"], reason: row.reason == null ? null : String(row.reason),
            answer: row.answer == null ? null : String(row.answer), createdAt: Number(row.created_at), updatedAt: Number(row.updated_at) }));
    }

    async close(): Promise<void> {
        if (this.initialization) await this.initialization;
        this.client?.close();
        this.client = null;
        this.initialization = null;
    }
}

export const taskStore = new TaskStore();

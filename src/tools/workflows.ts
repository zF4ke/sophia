import { z } from "zod";
import { T } from "@/shared/discordTools";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import type { ToolDefinition } from "./types";

async function ensureTable() {
    await OperationalStore.initialize();
    const c = OperationalStore.getClient();
    await c.executeMultiple(`
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            guild_id TEXT,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            steps_json TEXT NOT NULL,
            created_by TEXT,
            created_timestamp INTEGER NOT NULL,
            updated_timestamp INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_workflows_guild ON workflows(guild_id);
    `);
}
function now() { return Date.now(); }
function uid() { return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`; }

type WorkflowStep = { tool: string; args: Record<string, unknown>; label?: string };

const createParams = {
    type: "object",
    properties: {
        name: { type: "string", description: "Workflow name (unique per guild, slug-like, e.g. 'weekly-report')." },
        description: { type: "string", description: "Human description of what this workflow does." },
        steps: {
            type: "array",
            description: "Ordered steps: each { tool: string, args: object, label?: string }.",
            items: {
                type: "object",
                properties: {
                    tool: { type: "string", description: "Tool name to call." },
                    args: { type: "object", description: "Arguments for the tool.", properties: {}, required: [] },
                    label: { type: "string", description: "Optional label for this step." },
                },
                required: ["tool", "args"],
            },
        },
    },
    required: ["name", "description", "steps"],
} as const;

export const workflowCreateTool: ToolDefinition = {
    name: T.workflow_create,
    catalog: { effect: "write", description: "Create a reusable workflow (sequence of tool calls) for the guild.", evidenceRole: "discovery_only" },
    schema: {
        description: "Create a workflow that chains multiple tools. Example: weekly report = list_guild_structure + retrieve_messages + note_add. Workflows are stored per guild and can be run via workflow_run.",
        parameters: createParams,
    },
    capability: {
        description: "Create workflow.",
        inputSchema: z.object({
            name: z.string().min(2).max(48),
            description: z.string().min(5).max(500),
            steps: z.array(z.object({ tool: z.string(), args: z.record(z.string(), z.unknown()), label: z.string().optional() })).min(1).max(20),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["workflow stored"],
        async run(context, args) {
            await ensureTable();
            const name = String(args.name || "").trim().toLowerCase().replace(/[^a-z0-9_-]/g, "-");
            const description = String(args.description || "").trim();
            const steps = args.steps as WorkflowStep[];
            if (!name || !description || !Array.isArray(steps) || steps.length === 0) {
                return { tool: T.workflow_create, summary: "Invalid workflow.", data: null, errorMessage: "name, description, steps required." };
            }
            const guildId = context.guild?.id ?? null;
            const c = OperationalStore.getClient();
            const exists = await c.execute({ sql: `SELECT id FROM workflows WHERE guild_id = :guildId AND name = :name`, args: { guildId, name } });
            if (exists.rows.length) return { tool: T.workflow_create, summary: `Workflow "${name}" already exists.`, data: null, errorMessage: "Name already exists. Use a different name or delete first." };
            const id = uid();
            const ts = now();
            await c.execute({
                sql: `INSERT INTO workflows (id, guild_id, name, description, steps_json, created_by, created_timestamp, updated_timestamp) VALUES (:id,:guildId,:name,:description,:stepsJson,:createdBy,:ts,:ts)`,
                args: { id, guildId, name, description, stepsJson: JSON.stringify(steps), createdBy: context.actorId ?? null, ts },
            });
            return { tool: T.workflow_create, summary: `Created workflow "${name}" with ${steps.length} steps.`, data: { id, name, guildId, steps } };
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "⚙️", labelPt: "Criar workflow" },
};

const listParams = { type: "object", properties: {}, required: [] } as const;

export const workflowListTool: ToolDefinition = {
    name: T.workflow_list,
    catalog: { effect: "read", description: "List workflows for the current guild.", evidenceRole: "discovery_only" },
    schema: { description: "List saved workflows for this server. See names, descriptions and step counts.", parameters: listParams },
    capability: {
        description: "List workflows.",
        inputSchema: z.object({}),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["returns workflow list"],
        async run(context) {
            await ensureTable();
            const guildId = context.guild?.id ?? null;
            const c = OperationalStore.getClient();
            const r = await c.execute({ sql: `SELECT id, name, description, steps_json, updated_timestamp FROM workflows WHERE guild_id = :guildId ORDER BY updated_timestamp DESC`, args: { guildId } });
            const workflows = r.rows.map((row) => {
                const rec = row as Record<string, unknown>;
                let steps: WorkflowStep[] = [];
                try { steps = JSON.parse(String(rec.steps_json)); } catch { steps = []; }
                return { id: String(rec.id), name: String(rec.name), description: String(rec.description), steps, stepCount: steps.length };
            });
            if (!workflows.length) return { tool: T.workflow_list, summary: "No workflows yet. Create one with workflow_create.", data: { workflows: [] } };
            const summary = workflows.map((w) => `• ${w.name}: ${w.description} (${w.stepCount} steps)`).join("\n");
            return { tool: T.workflow_list, summary: `Workflows (${workflows.length}):\n${summary}`, data: { workflows } };
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "📋", labelPt: "Listar workflows" },
};

const runParams = {
    type: "object",
    properties: {
        name: { type: "string", description: "Workflow name to run." },
        overrides: { type: "object", description: "Optional overrides for step args, keyed by step index (e.g. {'0': { query: 'new query' }}).", properties: {}, required: [] },
    },
    required: ["name"],
} as const;

const deleteParams = {
    type: "object",
    properties: {
        name: { type: "string", description: "Workflow name to delete." },
    },
    required: ["name"],
} as const;

export const workflowDeleteTool: ToolDefinition = {
    name: T.workflow_delete,
    catalog: { effect: "write", description: "Delete a saved workflow for the current guild.", evidenceRole: "discovery_only" },
    schema: {
        description: "Delete a saved workflow by name. Use when the user asks to remove a workflow.",
        parameters: deleteParams,
    },
    capability: {
        description: "Delete workflow.",
        inputSchema: z.object({ name: z.string().min(1) }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["workflow removed when it existed"],
        async run(context, args) {
            await ensureTable();
            const name = String(args.name || "").trim().toLowerCase();
            const guildId = context.guild?.id ?? null;
            const c = OperationalStore.getClient();
            const result = await c.execute({ sql: `DELETE FROM workflows WHERE guild_id = :guildId AND name = :name`, args: { guildId, name } });
            const removed = Number(result.rowsAffected ?? 0);
            if (!removed) return { tool: T.workflow_delete, summary: `Workflow "${name}" not found.`, data: { name, removed: false } };
            return { tool: T.workflow_delete, summary: `Deleted workflow "${name}".`, data: { name, removed: true } };
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "🗑️", labelPt: "Apagar workflow" },
};

export const workflowRunTool: ToolDefinition = {
    name: T.workflow_run,
    catalog: { effect: "read", description: "Load a saved workflow so its steps can run through the normal runtime executor.", evidenceRole: "discovery_only" },
    schema: {
        description: "Load a saved workflow by name. Then call each returned step through the normal tool interface, in order. This keeps budgets, tracing, and approvals intact.",
        parameters: runParams,
    },
    capability: {
        description: "Run workflow.",
        inputSchema: z.object({ name: z.string().min(1), overrides: z.record(z.string(), z.unknown()).optional() }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: [],
        postconditions: ["returns the workflow steps for normal runtime execution"],
        async run(context, args) {
            await ensureTable();
            const name = String(args.name || "").trim().toLowerCase();
            const guildId = context.guild?.id ?? null;
            const c = OperationalStore.getClient();
            const row = (await c.execute({ sql: `SELECT * FROM workflows WHERE guild_id = :guildId AND name = :name`, args: { guildId, name } })).rows[0] as Record<string, unknown> | undefined;
            if (!row) return { tool: T.workflow_run, summary: `Workflow "${name}" not found.`, data: null, errorMessage: `No workflow named "${name}". List with workflow_list.` };
            let steps: WorkflowStep[] = [];
            try { steps = JSON.parse(String(row.steps_json)); } catch { steps = []; }
            const overrides = (args.overrides ?? {}) as Record<string, Record<string, unknown>>;
            const preparedSteps = steps.map((step, index) => {
                const args = { ...(step.args ?? {}) } as Record<string, unknown>;
                const override = overrides[String(index)];
                if (override && typeof override === "object") Object.assign(args, override);
                return { index, tool: String(step.tool), args, label: step.label ?? null };
            });
            const summary = preparedSteps
                .map((step) => `[${step.index}] ${step.tool}${step.label ? ` (${step.label})` : ""}`)
                .join("\n");
            return {
                tool: T.workflow_run,
                summary: `Loaded workflow "${name}" with ${preparedSteps.length} step(s). Execute these through normal tool calls:\n${summary}`,
                data: { name, steps: preparedSteps, executionRequired: true },
            };
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "▶️", labelPt: "Executar workflow" },
};

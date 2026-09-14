import { z } from "zod";
import { T } from "@/shared/discordTools";
import { getWorkingState } from "@/runtime/tasks/workingState";
import type { ToolDefinition, CapabilityContext } from "./types";

const MAX_BODY_CHARS = 4000;
const MAX_LABEL_CHARS = 64;

function requireIds(context: CapabilityContext): { requestId: string; threadId: string } | { error: string } {
    const requestId = context.requestId ?? null;
    const threadId = context.threadId ?? null;
    if (!requestId || !threadId) {
        return { error: "Scratchpad unavailable: runtime request context is missing." };
    }
    return { requestId, threadId };
}


// ── note_add ────────────────────────────────────────────────────────

const noteAddParameters = {
    type: "object",
    properties: {
        body: {
            type: "string",
            description:
                "Short finding, 2–5 bullet lines preferred. Reference messages by jumpLink rather than pasting raw content. Max ~4000 chars.",
        },
        label: {
            type: "string",
            description:
                "Optional category tag for filtering later (e.g. 'toxicity', 'links', 'decisions'). Keep short and lowercase.",
        },
    },
    required: ["body"],
} as const;

export const noteAddTool: ToolDefinition = {
    name: T.note_add,
    catalog: {
        effect: "read",
        description:
            "Append a note to this task's scratchpad. Use during long scans to record partial findings so they survive context compaction.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Append a finding to this task's scratchpad. Prefer short bullets with jumpLinks over raw quotes. Notes persist across continuation turns and restarts. Use note_list to read the current task notes.",
        parameters: noteAddParameters,
    },
    capability: {
        description: "Append a note to the task scratchpad.",
        inputSchema: z.object({
            body: z.string(),
            label: z.string().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["one note saved in the active working state"],
        async run(context, args) {
            const state = await getWorkingState(context);
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.note_add, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const body = String(args.body ?? "").slice(0, MAX_BODY_CHARS).trim();
            if (!body) {
                return { tool: T.note_add, summary: "Empty note rejected.", data: null, errorMessage: "empty_body" };
            }
            const label = args.label == null ? null : String(args.label).slice(0, MAX_LABEL_CHARS).trim() || null;

            const count = await state.countRequestNotes({
                requestId: ids.requestId,
                kind: "note",
            });

            const { seq } = await state.addRequestNote({
                requestId: ids.requestId,
                threadId: ids.threadId,
                kind: "note",
                label,
                body,
            });
            return {
                tool: T.note_add,
                summary: `Note #${seq} saved${label ? ` [${label}]` : ""} (${count + 1} notes).`,
                data: { seq, label, count: count + 1, cap: null },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "📝", labelPt: "Anotar" },
};

// ── note_list ───────────────────────────────────────────────────────

const noteListParameters = {
    type: "object",
    properties: {
        include_thread_history: {
            type: "boolean",
            description:
                "Compatibility option. Notes always cover the active task across continuation turns, never other tasks in the channel.",
        },
        label: {
            type: "string",
            description: "Optional: return only notes with this exact label.",
        },
    },
    required: [],
} as const;

export const noteListTool: ToolDefinition = {
    name: T.note_list,
    catalog: {
        effect: "read",
        description: "Read back this task's notes, plan and goals.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "List the active task notes across all continuation turns. Other tasks are excluded. Output is ordered by creation time.",
        parameters: noteListParameters,
    },
    capability: {
        description: "List request scratchpad notes.",
        inputSchema: z.object({
            include_thread_history: z.boolean().optional(),
            label: z.string().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["returns the active task's working state"],
        async run(context, args) {
            const state = await getWorkingState(context);
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.note_list, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const includeThreadHistory = Boolean(args.include_thread_history);
            const label = args.label == null ? undefined : String(args.label).trim() || undefined;

            const notes = await state.listRequestNotes({
                requestId: ids.requestId,
                threadId: ids.threadId,
                includeThreadHistory,
                kind: "note",
                label,
            });
            const planBody = await state.getRequestPlan(ids.requestId);
            const goals = context.taskId ? await state.listRequestGoals({ requestId: ids.requestId }) : [];

            const summary =
                `${notes.length} note(s)` +
                (planBody ? " + plan" : "") +
                (includeThreadHistory && !context.taskId ? " (incl. thread history)" : "");

            return {
                tool: T.note_list,
                summary,
                data: {
                    plan: planBody,
                    ...(context.taskId ? { goals: goals.map(goal => ({ seq: goal.seq, label: goal.label, body: goal.body, status: goal.status })) } : {}),
                    notes: notes.map((n) => ({
                        seq: n.seq,
                        label: n.label,
                        body: n.body,
                        created_at: n.createdTimestamp,
                        request_id: n.requestId,
                    })),
                },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "📖", labelPt: "Ler anotações" },
};

// ── note_clear ──────────────────────────────────────────────────────

const noteClearParameters = {
    type: "object",
    properties: {
        label: {
            type: "string",
            description: "Optional: clear only notes with this label. Omit to clear every note for this task.",
        },
    },
    required: [],
} as const;

export const noteClearTool: ToolDefinition = {
    name: T.note_clear,
    catalog: {
        effect: "read",
        description: "Clear notes from the task scratchpad.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Delete notes from the current request's scratchpad. Optional label filter restricts deletion to a subset. Does not affect the plan — use plan_update to overwrite that.",
        parameters: noteClearParameters,
    },
    capability: {
        description: "Clear scratchpad notes for the current request.",
        inputSchema: z.object({ label: z.string().optional() }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["matching notes removed for current request"],
        async run(context, args) {
            const state = await getWorkingState(context);
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.note_clear, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const label = args.label == null ? undefined : String(args.label).trim() || undefined;
            const { removed } = await state.clearRequestNotes({
                requestId: ids.requestId,
                kind: "note",
                label,
            });
            return {
                tool: T.note_clear,
                summary: `Removed ${removed} note(s)${label ? ` with label "${label}"` : ""}.`,
                data: { removed, label: label ?? null },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "🧹", labelPt: "Limpar anotações" },
};

// ── plan_update ─────────────────────────────────────────────────────

const planUpdateParameters = {
    type: "object",
    properties: {
        body: {
            type: "string",
            description:
                "Full plan text. Suggested sections: Goal, Approach, Progress. Overwrites the previous plan for this task.",
        },
    },
    required: ["body"],
} as const;

export const planUpdateTool: ToolDefinition = {
    name: T.plan_update,
    catalog: {
        effect: "read",
        description: "Write or overwrite the plan for this task. Injected into the system prompt every turn.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Write the plan for this task. Overwrites any previous plan. The runtime reloads it as working context on every subsequent model call, so it survives compaction. Record the objective, remaining steps and completed work.",
        parameters: planUpdateParameters,
    },
    capability: {
        description: "Upsert the task plan.",
        inputSchema: z.object({ body: z.string() }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["plan row upserted for current request"],
        async run(context, args) {
            const state = await getWorkingState(context);
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.plan_update, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const body = String(args.body ?? "").slice(0, MAX_BODY_CHARS * 2).trim();
            if (!body) {
                return { tool: T.plan_update, summary: "Empty plan rejected.", data: null, errorMessage: "empty_body" };
            }
            const { version } = await state.upsertRequestPlan({
                requestId: ids.requestId,
                threadId: ids.threadId,
                body,
            });
            return {
                tool: T.plan_update,
                summary: `Plan updated (v${version}).`,
                data: { version },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "🗺️", labelPt: "Atualizar plano" },
};

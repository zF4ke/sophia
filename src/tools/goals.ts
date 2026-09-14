import { z } from "zod";
import { T } from "@/shared/discordTools";
import { getWorkingState } from "@/runtime/tasks/workingState";
import { SettingsService } from "@/app/SettingsService";
import type { ToolDefinition, CapabilityContext } from "./types";

export const GOAL_STATUSES = ["open", "in_progress", "blocked", "done", "cancelled"] as const;
export type GoalStatus = (typeof GOAL_STATUSES)[number];

const MAX_BODY_CHARS = 4000;
const MAX_LABEL_CHARS = 64;
const DEFAULT_MAX_GOALS = 10;

function requireIds(context: CapabilityContext): { requestId: string; threadId: string } | { error: string } {
    const requestId = context.requestId ?? null;
    const threadId = context.threadId ?? null;
    if (!requestId || !threadId) {
        return { error: "Goals unavailable: runtime request context is missing." };
    }
    return { requestId, threadId };
}

function settingMaxGoals(): number {
    const s = SettingsService.load();
    const value = (s.runtime as unknown as { maxGoalsPerRequest?: number }).maxGoalsPerRequest;
    return typeof value === "number" && value > 0 ? value : DEFAULT_MAX_GOALS;
}

/** Collapse a goal body to its essence for twin detection. */
function normalizeGoalBody(body: string): string {
    return body
        .toLowerCase()
        .normalize("NFD")
        .replace(/[̀-ͯ]/g, "")
        .replace(/[^a-z0-9\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

const openParameters = {
    type: "object",
    properties: {
        body: {
            type: "string",
            description:
                "The user-facing task in one sentence. Write it as the user asked (e.g. 'Gerir newsletter semanal: pesquisar, redigir e enviar para #novidades'). Max ~4000 chars.",
        },
        label: {
            type: "string",
            description:
                "Optional short tag for filtering later (e.g. 'newsletter', 'cleanup'). Keep short and lowercase.",
        },
    },
    required: ["body"],
} as const;

export const goalOpenTool: ToolDefinition = {
    name: T.goal_open,
    catalog: {
        effect: "read",
        description:
            "Open a persistent goal for the current task. Goals survive across turns; the runtime chains new turns until every goal is done, blocked, or cancelled.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Open a goal describing the task you are starting. Goals persist for the whole task and the runtime keeps chaining turns until each goal is done, blocked, or cancelled. Open one goal per distinct user-facing task, never per turn or per step.",
        parameters: openParameters,
    },
    capability: {
        description: "Open a persistent goal for the current task.",
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
        postconditions: ["one goal saved in the active working state"],
        async run(context, args) {
            const state = await getWorkingState(context);
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.goal_open, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const body = String(args.body ?? "").slice(0, MAX_BODY_CHARS).trim();
            if (!body) {
                return { tool: T.goal_open, summary: "Empty goal rejected.", data: null, errorMessage: "empty_body" };
            }
            const label = args.label == null ? null : String(args.label).slice(0, MAX_LABEL_CHARS).trim() || null;

            const cap = settingMaxGoals();
            const goals = await state.listRequestGoals({
                requestId: ids.requestId,
                threadId: ids.threadId,
                includeThreadHistory: true,
            });
            if (!context.taskId && goals.filter((goal) => goal.status !== "done" && goal.status !== "cancelled").length >= cap) {
                return {
                    tool: T.goal_open,
                    summary: `Goal limit reached (${cap}). Mark obsolete goals done or cancelled before opening new ones.`,
                    data: { error: "goal_limit", cap },
                    errorMessage: "goal_limit",
                };
            }

            // Twin guard: an open goal with a near-identical body means the
            // model opened the same task twice (common on chained legs). Reuse
            // the existing row instead of duplicating it — duplicates are what
            // used to print the same objective twice in every status message.
            const normalized = normalizeGoalBody(body);
            const candidateWords = new Set(normalized.split(" ").filter((word) => word.length > 2));
            const twin = goals.find((goal) => {
                if (goal.status !== "open" && goal.status !== "in_progress") return false;
                if (context.taskId) return goal.body.trim() === body;
                if (normalizeGoalBody(goal.body) === normalized) return true;
                const kept = new Set(normalizeGoalBody(goal.body).split(" ").filter((word) => word.length > 2));
                let intersection = 0;
                for (const word of candidateWords) if (kept.has(word)) intersection++;
                const union = candidateWords.size + kept.size - intersection;
                const jaccard = union === 0 ? 0 : intersection / union;
                return jaccard > 0.55;
            });
            if (twin) {
                return {
                    tool: T.goal_open,
                    summary: `Goal #${twin.seq} already covers this task; reusing it instead of opening a duplicate.`,
                    data: { seq: twin.seq, label: twin.label, reused: true },
                };
            }

            const { seq } = await state.addRequestGoal({
                requestId: ids.requestId,
                threadId: ids.threadId,
                label,
                body,
            });
            return {
                tool: T.goal_open,
                summary: `Goal #${seq} opened${label ? ` [${label}]` : ""}.`,
                data: { seq, label },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "🎯", labelPt: "Abrir objetivo" },
};

// ── goal_update ─────────────────────────────────────────────────────

const updateParameters = {
    type: "object",
    properties: {
        seq: {
            type: "number",
            description: "Goal sequence number from goal_open.",
        },
        status: {
            type: "string",
            description: `New status: open, in_progress, blocked, done, or cancelled.`,
        },
        body: {
            type: "string",
            description: "Optional replacement body when the task was reframed.",
        },
    },
    required: ["seq", "status"],
} as const;

export const goalUpdateTool: ToolDefinition = {
    name: T.goal_update,
    catalog: {
        effect: "read",
        description: "Move a goal between open, in_progress, blocked, done, and cancelled.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Update a goal's status. Mark in_progress when started, blocked when waiting on the user or an approval, done when the user-facing result was delivered. Done and cancelled goals stop the auto-continue chain.",
        parameters: updateParameters,
    },
    capability: {
        description: "Update a goal's status (open/in_progress/blocked/done/cancelled).",
        inputSchema: z.object({
            seq: z.number(),
            status: z.enum(GOAL_STATUSES),
            body: z.string().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["goal row updated when it existed"],
        async run(context, args) {
            const state = await getWorkingState(context);
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.goal_update, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const seq = Math.floor(Number(args.seq));
            if (!Number.isFinite(seq) || seq <= 0) {
                return { tool: T.goal_update, summary: "Invalid goal seq.", data: null, errorMessage: "invalid_seq" };
            }

            const goals = await state.listRequestGoals({
                requestId: ids.requestId,
                threadId: ids.threadId,
                includeThreadHistory: true,
            });
            const target = goals.find((goal) => goal.seq === seq);
            if (!target) {
                return { tool: T.goal_update, summary: `Goal #${seq} not found in this task.`, data: { error: "not_found", seq }, errorMessage: "not_found" };
            }
            const status = String(args.status) as GoalStatus;
            const body = args.body == null ? undefined : String(args.body).slice(0, MAX_BODY_CHARS).trim() || undefined;
            await state.updateRequestGoal({
                requestId: target.requestId,
                seq,
                status,
                body,
            });
            return {
                tool: T.goal_update,
                summary: `Goal #${seq} → ${status}.`,
                data: { seq, status },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "🔄", labelPt: "Atualizar objetivo" },
};

// ── goal_done ───────────────────────────────────────────────────────

const doneParameters = {
    type: "object",
    properties: {
        seq: {
            type: "number",
            description: "Goal sequence number from goal_open.",
        },
        note: {
            type: "string",
            description: "Optional one-line summary of the delivered result for the next turn to read.",
        },
    },
    required: ["seq"],
} as const;

export const goalDoneTool: ToolDefinition = {
    name: T.goal_done,
    catalog: {
        effect: "read",
        description: "Mark a goal done. Only call when the user-facing result was delivered, not when work merely started.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Close a goal as done after delivering the user-facing result. Optionally leave a one-line note for the audit trail. A goal is done only when the user asked for something and got it — not when you began exploring it.",
        parameters: doneParameters,
    },
    capability: {
        description: "Mark a goal done with an optional result note.",
        inputSchema: z.object({
            seq: z.number(),
            note: z.string().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["goal row marked done when it existed"],
        async run(context, args) {
            const state = await getWorkingState(context);
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.goal_done, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const seq = Math.floor(Number(args.seq));
            if (!Number.isFinite(seq) || seq <= 0) {
                return { tool: T.goal_done, summary: "Invalid goal seq.", data: null, errorMessage: "invalid_seq" };
            }
            const goals = await state.listRequestGoals({
                requestId: ids.requestId,
                threadId: ids.threadId,
                includeThreadHistory: true,
            });
            const target = goals.find((goal) => goal.seq === seq);
            if (!target) {
                return { tool: T.goal_done, summary: `Goal #${seq} not found in this task.`, data: { error: "not_found", seq }, errorMessage: "not_found" };
            }
            const note = args.note == null ? null : String(args.note).slice(0, MAX_BODY_CHARS).trim() || null;
            await state.updateRequestGoal({
                requestId: target.requestId,
                seq,
                status: "done",
                body: note ? `${target.body}\n— ${note}` : undefined,
            });
            return {
                tool: T.goal_done,
                summary: `Goal #${seq} done${note ? `: ${note}` : "."}`,
                data: { seq, status: "done" },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "✅", labelPt: "Concluir objetivo" },
};

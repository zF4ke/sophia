import { z } from "zod";
import { T } from "@/shared/discordTools";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { SettingsService } from "@/app/SettingsService";
import type { ToolDefinition, CapabilityContext } from "./types";

const MAX_BODY_CHARS = 4000;
const MAX_LABEL_CHARS = 64;
const MAX_PREVIEW_CHARS = 80;
const DEFAULT_MAX_NOTES = 200;

function requireIds(context: CapabilityContext): { requestId: string; threadId: string } | { error: string } {
    const requestId = context.requestId ?? null;
    const threadId = context.threadId ?? null;
    if (!requestId || !threadId) {
        return { error: "Scratchpad unavailable: runtime request context is missing." };
    }
    return { requestId, threadId };
}

function settingMaxNotes(): number {
    const s = SettingsService.load();
    const value = (s.runtime as unknown as { maxNotesPerRequest?: number }).maxNotesPerRequest;
    return typeof value === "number" && value > 0 ? value : DEFAULT_MAX_NOTES;
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
            "Append a note to this turn's scratchpad. Use during long scans to record partial findings so they survive context compaction.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Append a finding to this turn's per-request scratchpad. Prefer short bullets with jumpLinks over raw quotes. Notes persist across tool calls within the same turn and can be listed with note_list. Notes are isolated per request by default.",
        parameters: noteAddParameters,
    },
    capability: {
        description: "Append a note to the per-request scratchpad.",
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
        postconditions: ["one row appended to request_notes"],
        async run(context, args) {
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.note_add, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const body = String(args.body ?? "").slice(0, MAX_BODY_CHARS).trim();
            if (!body) {
                return { tool: T.note_add, summary: "Empty note rejected.", data: null, errorMessage: "empty_body" };
            }
            const label = args.label == null ? null : String(args.label).slice(0, MAX_LABEL_CHARS).trim() || null;

            const cap = settingMaxNotes();
            const count = await DiscordMemoryService.countRequestNotes({
                requestId: ids.requestId,
                kind: "note",
            });
            if (count >= cap) {
                return {
                    tool: T.note_add,
                    summary: `Note limit reached (${cap}). Use note_list to review or note_clear to prune.`,
                    data: { error: "note_limit", cap },
                    errorMessage: "note_limit",
                };
            }

            const { seq } = await DiscordMemoryService.addRequestNote({
                requestId: ids.requestId,
                threadId: ids.threadId,
                kind: "note",
                label,
                body,
            });
            return {
                tool: T.note_add,
                summary: `Note #${seq} saved${label ? ` [${label}]` : ""} (${count + 1}/${cap}).`,
                data: { seq, label, count: count + 1, cap },
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
                "When true, also include notes from earlier completed requests on the same thread. Default false (current request only).",
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
        description: "Browse the notebook — returns metadata and previews, not full contents.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Browse notebook pages. Returns note IDs, labels, short previews and timestamps — not full bodies. Use note_read to open a specific page. Pass include_thread_history: true to also see notes from earlier turns on the same thread.",
        parameters: noteListParameters,
    },
    capability: {
        description: "Browse notebook pages (metadata only).",
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
        postconditions: ["returns note metadata for current request (and thread, if opted in)"],
        async run(context, args) {
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.note_list, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const includeThreadHistory = Boolean(args.include_thread_history);
            const label = args.label == null ? undefined : String(args.label).trim() || undefined;

            const notes = await DiscordMemoryService.listRequestNotes({
                requestId: ids.requestId,
                threadId: ids.threadId,
                includeThreadHistory,
                kind: "note",
                label,
            });
            const planBody = await DiscordMemoryService.getRequestPlan(ids.requestId);

            const summary =
                `${notes.length} page(s)` +
                (planBody ? " + plan" : "") +
                (includeThreadHistory ? " (incl. thread history)" : "");

            return {
                tool: T.note_list,
                summary,
                data: {
                    plan: planBody,
                    notes: notes.map((n) => ({
                        seq: n.seq,
                        label: n.label,
                        preview: n.body.length > MAX_PREVIEW_CHARS
                            ? n.body.slice(0, MAX_PREVIEW_CHARS) + "…"
                            : n.body,
                        created_at: n.createdTimestamp,
                        request_id: n.requestId,
                    })),
                },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "📖", labelPt: "Folhear caderno" },
};

// ── note_read ───────────────────────────────────────────────────────

const noteReadParameters = {
    type: "object",
    properties: {
        seq: {
            type: "number",
            description: "Read a single note by its sequence number.",
        },
        seqs: {
            type: "array",
            items: { type: "number" },
            description: "Read multiple notes by sequence numbers.",
        },
        label: {
            type: "string",
            description: "Read all notes with this exact label.",
        },
    },
    required: [],
} as const;

export const noteReadTool: ToolDefinition = {
    name: T.note_read,
    catalog: {
        effect: "read",
        description: "Open notebook pages and return full contents.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Open one or more notebook pages and return their full contents. Filter by seq, seqs (array), or label. If none provided, returns all notes for the current request.",
        parameters: noteReadParameters,
    },
    capability: {
        description: "Read full contents of notebook pages.",
        inputSchema: z.object({
            seq: z.number().optional(),
            seqs: z.array(z.number()).optional(),
            label: z.string().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["returns full note bodies for matching notes"],
        async run(context, args) {
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.note_read, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const label = args.label == null ? undefined : String(args.label).trim() || undefined;

            const allNotes = await DiscordMemoryService.listRequestNotes({
                requestId: ids.requestId,
                threadId: ids.threadId,
                kind: "note",
                label,
            });

            let filtered = allNotes;
            const singleSeq = args.seq != null ? Number(args.seq) : null;
            const multiSeq = Array.isArray(args.seqs) ? args.seqs.map(Number) : null;

            if (singleSeq != null) {
                filtered = allNotes.filter((n) => n.seq === singleSeq);
            } else if (multiSeq != null && multiSeq.length > 0) {
                const seqSet = new Set(multiSeq);
                filtered = allNotes.filter((n) => seqSet.has(n.seq));
            }

            return {
                tool: T.note_read,
                summary: `${filtered.length} page(s) returned.`,
                data: {
                    notes: filtered.map((n) => ({
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
    display: { icon: "📄", labelPt: "Ler página" },
};

// ── note_update ─────────────────────────────────────────────────────

const noteUpdateParameters = {
    type: "object",
    properties: {
        seq: {
            type: "number",
            description: "Sequence number of the note to rewrite.",
        },
        body: {
            type: "string",
            description: "New full content for the note (replaces previous body entirely). Max ~4000 chars.",
        },
        label: {
            type: "string",
            description: "Optional: update the label. Omit to keep the existing label.",
        },
    },
    required: ["seq", "body"],
} as const;

export const noteUpdateTool: ToolDefinition = {
    name: T.note_update,
    catalog: {
        effect: "read",
        description: "Rewrite a notebook page in place.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Rewrite a notebook page by sequence number. Full replace — the old body is overwritten entirely. Use this to refine, correct, or consolidate an existing note instead of creating duplicates.",
        parameters: noteUpdateParameters,
    },
    capability: {
        description: "Rewrite a notebook page in place.",
        inputSchema: z.object({
            seq: z.number(),
            body: z.string(),
            label: z.string().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["note body replaced for matching seq"],
        async run(context, args) {
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.note_update, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const seq = Number(args.seq);
            if (!Number.isFinite(seq) || seq < 1) {
                return { tool: T.note_update, summary: "Invalid seq.", data: null, errorMessage: "invalid_seq" };
            }
            const body = String(args.body ?? "").slice(0, MAX_BODY_CHARS).trim();
            if (!body) {
                return { tool: T.note_update, summary: "Empty body rejected.", data: null, errorMessage: "empty_body" };
            }
            const label = args.label === undefined ? undefined : (args.label == null ? null : String(args.label).slice(0, MAX_LABEL_CHARS).trim() || null);

            const { updated } = await DiscordMemoryService.updateRequestNote({
                requestId: ids.requestId,
                seq,
                body,
                label,
            });
            if (!updated) {
                return {
                    tool: T.note_update,
                    summary: `Note #${seq} not found.`,
                    data: { seq, updated: false },
                    errorMessage: "not_found",
                };
            }
            return {
                tool: T.note_update,
                summary: `Note #${seq} rewritten${label !== undefined ? ` [${label ?? "unlabelled"}]` : ""}.`,
                data: { seq, updated: true },
            };
        },
    },
    strategy: { extractEvidence: () => [] },
    display: { icon: "✏️", labelPt: "Reescrever página" },
};

// ── note_clear ──────────────────────────────────────────────────────

const noteClearParameters = {
    type: "object",
    properties: {
        label: {
            type: "string",
            description: "Optional: clear only notes with this label. Omit to clear every note for this request.",
        },
    },
    required: [],
} as const;

export const noteClearTool: ToolDefinition = {
    name: T.note_clear,
    catalog: {
        effect: "read",
        description: "Clear notes from the per-request scratchpad.",
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
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.note_clear, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const label = args.label == null ? undefined : String(args.label).trim() || undefined;
            const { removed } = await DiscordMemoryService.clearRequestNotes({
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
                "Full plan text. Suggested sections: Goal, Approach, Progress. Overwrites the previous plan for this request.",
        },
    },
    required: ["body"],
} as const;

export const planUpdateTool: ToolDefinition = {
    name: T.plan_update,
    catalog: {
        effect: "read",
        description: "Write or overwrite the plan for this request. Injected into the system prompt every turn.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Write the plan for this request. Overwrites any previous plan. The plan is prepended to the system prompt on every subsequent tool-call round, so it survives compaction. Use it for Goal / Approach / Progress.",
        parameters: planUpdateParameters,
    },
    capability: {
        description: "Upsert the per-request plan.",
        inputSchema: z.object({ body: z.string() }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["plan row upserted for current request"],
        async run(context, args) {
            const ids = requireIds(context);
            if ("error" in ids) {
                return { tool: T.plan_update, summary: ids.error, data: null, errorMessage: ids.error };
            }
            const body = String(args.body ?? "").slice(0, MAX_BODY_CHARS * 2).trim();
            if (!body) {
                return { tool: T.plan_update, summary: "Empty plan rejected.", data: null, errorMessage: "empty_body" };
            }
            const { version } = await DiscordMemoryService.upsertRequestPlan({
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

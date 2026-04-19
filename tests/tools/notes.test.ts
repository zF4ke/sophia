import { describe, expect, it, vi, beforeEach } from "vitest";
import { noteAddTool, noteListTool, noteReadTool, noteUpdateTool, noteClearTool, planUpdateTool } from "@/tools/notes";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { CapabilityContext } from "@/tools/types";
import type { Guild } from "discord.js";

vi.mock("@/memory/DiscordMemoryService", () => ({
    DiscordMemoryService: {
        addRequestNote: vi.fn(),
        countRequestNotes: vi.fn(),
        listRequestNotes: vi.fn(),
        getRequestPlan: vi.fn(),
        clearRequestNotes: vi.fn(),
        upsertRequestPlan: vi.fn(),
        updateRequestNote: vi.fn(),
    },
}));

vi.mock("@/app/SettingsService", () => ({
    SettingsService: {
        load: () => ({
            runtime: { maxNotesPerRequest: 5 },
            compaction: { summarizerModel: "test", triggerFraction: 0.85 },
        }),
    },
}));

function ctx(overrides: Partial<CapabilityContext> = {}): CapabilityContext {
    return {
        guild: null as unknown as Guild,
        question: "test",
        requestId: "req-1",
        threadId: "thread-1",
        ...overrides,
    };
}

beforeEach(() => {
    vi.clearAllMocks();
});

// ── note_add ────────────────────────────────────────────────────────

describe("note_add", () => {
    const run = noteAddTool.capability.run.bind(null);

    it("saves a note and returns seq", async () => {
        (DiscordMemoryService.countRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue(0);
        (DiscordMemoryService.addRequestNote as ReturnType<typeof vi.fn>).mockResolvedValue({ seq: 1 });

        const result = await run(ctx(), { body: "Found something" });
        expect(result.data).toEqual({ seq: 1, label: null, count: 1, cap: 5 });
        expect(DiscordMemoryService.addRequestNote).toHaveBeenCalledWith(
            expect.objectContaining({ requestId: "req-1", kind: "note", body: "Found something" }),
        );
    });

    it("rejects empty body", async () => {
        const result = await run(ctx(), { body: "" });
        expect(result.errorMessage).toBe("empty_body");
    });

    it("rejects when note limit reached", async () => {
        (DiscordMemoryService.countRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue(5);

        const result = await run(ctx(), { body: "Over limit" });
        expect(result.errorMessage).toBe("note_limit");
    });

    it("errors when requestId is missing", async () => {
        const result = await run(ctx({ requestId: null }), { body: "no ctx" });
        expect(result.errorMessage).toContain("runtime request context");
    });

    it("truncates body to max chars", async () => {
        (DiscordMemoryService.countRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue(0);
        (DiscordMemoryService.addRequestNote as ReturnType<typeof vi.fn>).mockResolvedValue({ seq: 1 });

        const longBody = "x".repeat(5000);
        await run(ctx(), { body: longBody });
        const callArgs = (DiscordMemoryService.addRequestNote as ReturnType<typeof vi.fn>).mock.calls[0][0];
        expect(callArgs.body.length).toBeLessThanOrEqual(4000);
    });

    it("passes label when provided", async () => {
        (DiscordMemoryService.countRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue(0);
        (DiscordMemoryService.addRequestNote as ReturnType<typeof vi.fn>).mockResolvedValue({ seq: 2 });

        const result = await run(ctx(), { body: "tagged", label: "toxicity" });
        expect((result.data as Record<string, unknown>)?.label).toBe("toxicity");
    });
});

// ── note_list ───────────────────────────────────────────────────────

describe("note_list", () => {
    const run = noteListTool.capability.run.bind(null);

    it("returns previews and plan (not full bodies)", async () => {
        (DiscordMemoryService.listRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue([
            { seq: 1, label: null, body: "short note", createdTimestamp: 100, requestId: "req-1" },
            { seq: 2, label: "links", body: "x".repeat(200), createdTimestamp: 200, requestId: "req-1" },
        ]);
        (DiscordMemoryService.getRequestPlan as ReturnType<typeof vi.fn>).mockResolvedValue("my plan");

        const result = await run(ctx(), {});
        expect(result.summary).toContain("2 page(s)");
        expect(result.summary).toContain("+ plan");
        const data = result.data as Record<string, unknown>;
        expect(data.plan).toBe("my plan");
        const notes = data.notes as Array<Record<string, unknown>>;
        expect(notes).toHaveLength(2);
        // Short note: preview equals full body
        expect(notes[0].preview).toBe("short note");
        expect(notes[0]).not.toHaveProperty("body");
        // Long note: preview is truncated with ellipsis
        expect((notes[1].preview as string).length).toBeLessThanOrEqual(81); // 80 + "…"
        expect((notes[1].preview as string).endsWith("…")).toBe(true);
    });

    it("includes thread history when opted in", async () => {
        (DiscordMemoryService.listRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue([]);
        (DiscordMemoryService.getRequestPlan as ReturnType<typeof vi.fn>).mockResolvedValue(null);

        await run(ctx(), { include_thread_history: true });
        expect(DiscordMemoryService.listRequestNotes).toHaveBeenCalledWith(
            expect.objectContaining({ includeThreadHistory: true }),
        );
    });

    it("errors when requestId is missing", async () => {
        const result = await run(ctx({ requestId: null }), {});
        expect(result.errorMessage).toContain("runtime request context");
    });
});

// ── note_read ───────────────────────────────────────────────────────

describe("note_read", () => {
    const run = noteReadTool.capability.run.bind(null);

    it("returns full body for a single note by seq", async () => {
        (DiscordMemoryService.listRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue([
            { seq: 1, label: null, body: "full content here", createdTimestamp: 100, requestId: "req-1" },
            { seq: 2, label: null, body: "another note", createdTimestamp: 200, requestId: "req-1" },
        ]);

        const result = await run(ctx(), { seq: 1 });
        const notes = (result.data as Record<string, unknown>).notes as Array<Record<string, unknown>>;
        expect(notes).toHaveLength(1);
        expect(notes[0].body).toBe("full content here");
        expect(notes[0].seq).toBe(1);
    });

    it("returns multiple notes by seqs array", async () => {
        (DiscordMemoryService.listRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue([
            { seq: 1, label: null, body: "note1", createdTimestamp: 100, requestId: "req-1" },
            { seq: 2, label: null, body: "note2", createdTimestamp: 200, requestId: "req-1" },
            { seq: 3, label: null, body: "note3", createdTimestamp: 300, requestId: "req-1" },
        ]);

        const result = await run(ctx(), { seqs: [1, 3] });
        const notes = (result.data as Record<string, unknown>).notes as Array<Record<string, unknown>>;
        expect(notes).toHaveLength(2);
        expect(notes.map((n) => n.seq)).toEqual([1, 3]);
    });

    it("returns all notes when no filter provided", async () => {
        (DiscordMemoryService.listRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue([
            { seq: 1, label: null, body: "note1", createdTimestamp: 100, requestId: "req-1" },
        ]);

        const result = await run(ctx(), {});
        const notes = (result.data as Record<string, unknown>).notes as Array<Record<string, unknown>>;
        expect(notes).toHaveLength(1);
    });

    it("filters by label", async () => {
        (DiscordMemoryService.listRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue([
            { seq: 1, label: "links", body: "link note", createdTimestamp: 100, requestId: "req-1" },
        ]);

        const result = await run(ctx(), { label: "links" });
        expect(DiscordMemoryService.listRequestNotes).toHaveBeenCalledWith(
            expect.objectContaining({ label: "links" }),
        );
        const notes = (result.data as Record<string, unknown>).notes as Array<Record<string, unknown>>;
        expect(notes).toHaveLength(1);
    });

    it("errors when requestId is missing", async () => {
        const result = await run(ctx({ requestId: null }), {});
        expect(result.errorMessage).toContain("runtime request context");
    });
});

// ── note_update ─────────────────────────────────────────────────────

describe("note_update", () => {
    const run = noteUpdateTool.capability.run.bind(null);

    it("rewrites a note and returns updated=true", async () => {
        (DiscordMemoryService.updateRequestNote as ReturnType<typeof vi.fn>).mockResolvedValue({ updated: true });

        const result = await run(ctx(), { seq: 1, body: "new content" });
        expect(result.summary).toContain("#1 rewritten");
        expect((result.data as Record<string, unknown>).updated).toBe(true);
        expect(DiscordMemoryService.updateRequestNote).toHaveBeenCalledWith(
            expect.objectContaining({ requestId: "req-1", seq: 1, body: "new content" }),
        );
    });

    it("returns not_found when seq does not exist", async () => {
        (DiscordMemoryService.updateRequestNote as ReturnType<typeof vi.fn>).mockResolvedValue({ updated: false });

        const result = await run(ctx(), { seq: 999, body: "new content" });
        expect(result.errorMessage).toBe("not_found");
    });

    it("rejects empty body", async () => {
        const result = await run(ctx(), { seq: 1, body: "" });
        expect(result.errorMessage).toBe("empty_body");
    });

    it("rejects invalid seq", async () => {
        const result = await run(ctx(), { seq: -1, body: "content" });
        expect(result.errorMessage).toBe("invalid_seq");
    });

    it("passes label when provided", async () => {
        (DiscordMemoryService.updateRequestNote as ReturnType<typeof vi.fn>).mockResolvedValue({ updated: true });

        await run(ctx(), { seq: 1, body: "content", label: "refined" });
        expect(DiscordMemoryService.updateRequestNote).toHaveBeenCalledWith(
            expect.objectContaining({ label: "refined" }),
        );
    });

    it("errors when requestId is missing", async () => {
        const result = await run(ctx({ requestId: null }), { seq: 1, body: "test" });
        expect(result.errorMessage).toContain("runtime request context");
    });
});

// ── note_clear ──────────────────────────────────────────────────────

describe("note_clear", () => {
    const run = noteClearTool.capability.run.bind(null);

    it("clears all notes for request", async () => {
        (DiscordMemoryService.clearRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue({ removed: 3 });

        const result = await run(ctx(), {});
        expect(result.summary).toContain("3 note(s)");
        expect(DiscordMemoryService.clearRequestNotes).toHaveBeenCalledWith(
            expect.objectContaining({ requestId: "req-1" }),
        );
    });

    it("filters by label", async () => {
        (DiscordMemoryService.clearRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue({ removed: 1 });

        const result = await run(ctx(), { label: "toxicity" });
        expect((result.data as Record<string, unknown>).label).toBe("toxicity");
        expect(DiscordMemoryService.clearRequestNotes).toHaveBeenCalledWith(
            expect.objectContaining({ label: "toxicity" }),
        );
    });
});

// ── plan_update ─────────────────────────────────────────────────────

describe("plan_update", () => {
    const run = planUpdateTool.capability.run.bind(null);

    it("upserts plan and returns version", async () => {
        (DiscordMemoryService.upsertRequestPlan as ReturnType<typeof vi.fn>).mockResolvedValue({ version: 2 });

        const result = await run(ctx(), { body: "Goal: scan\nApproach: batch" });
        expect(result.summary).toContain("v2");
        expect(DiscordMemoryService.upsertRequestPlan).toHaveBeenCalledWith(
            expect.objectContaining({ requestId: "req-1", body: "Goal: scan\nApproach: batch" }),
        );
    });

    it("rejects empty plan", async () => {
        const result = await run(ctx(), { body: "" });
        expect(result.errorMessage).toBe("empty_body");
    });

    it("errors when requestId is missing", async () => {
        const result = await run(ctx({ requestId: null }), { body: "plan" });
        expect(result.errorMessage).toContain("runtime request context");
    });
});

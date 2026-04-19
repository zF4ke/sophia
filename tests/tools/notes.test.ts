import { describe, expect, it, vi, beforeEach } from "vitest";
import { noteAddTool, noteListTool, noteClearTool, planUpdateTool } from "@/tools/notes";
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

    it("returns notes and plan", async () => {
        (DiscordMemoryService.listRequestNotes as ReturnType<typeof vi.fn>).mockResolvedValue([
            { seq: 1, label: null, body: "note1", createdTimestamp: 100, requestId: "req-1" },
        ]);
        (DiscordMemoryService.getRequestPlan as ReturnType<typeof vi.fn>).mockResolvedValue("my plan");

        const result = await run(ctx(), {});
        expect(result.summary).toContain("1 note(s)");
        expect(result.summary).toContain("+ plan");
        expect((result.data as Record<string, unknown>).notes).toHaveLength(1);
        expect((result.data as Record<string, unknown>).plan).toBe("my plan");
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

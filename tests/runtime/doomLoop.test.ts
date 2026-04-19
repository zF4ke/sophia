import { describe, expect, it } from "vitest";
import { DoomLoopDetector, ProgressTracker } from "@/runtime/stallGuard";

describe("DoomLoopDetector", () => {
    it("returns ok for non-repeating calls", () => {
        const d = new DoomLoopDetector();
        expect(d.recordCall("search_messages", "a").action).toBe("ok");
        expect(d.recordCall("retrieve_messages", "b").action).toBe("ok");
        expect(d.recordCall("search_messages", "c").action).toBe("ok");
    });

    it("nudges after repeatThreshold consecutive identical calls", () => {
        const d = new DoomLoopDetector(5, 3);
        d.recordCall("search_messages", "abc");
        d.recordCall("search_messages", "abc");
        const r = d.recordCall("search_messages", "abc");
        expect(r.action).toBe("nudge");
        expect(r.message).toContain("search_messages");
        expect(d.getNudgeCount()).toBe(1);
    });

    it("does not nudge when args differ", () => {
        const d = new DoomLoopDetector(5, 3);
        d.recordCall("search_messages", "a");
        d.recordCall("search_messages", "b");
        const r = d.recordCall("search_messages", "c");
        expect(r.action).toBe("ok");
    });

    it("force_finish on second nudge", () => {
        const d = new DoomLoopDetector(5, 3);
        // First triplet → nudge
        d.recordCall("search_messages", "x");
        d.recordCall("search_messages", "x");
        d.recordCall("search_messages", "x"); // nudge 1
        // Break the streak then repeat → second nudge → force_finish
        d.recordCall("retrieve_messages", "y");
        d.recordCall("search_messages", "x");
        d.recordCall("search_messages", "x");
        const r = d.recordCall("search_messages", "x"); // nudge 2
        expect(r.action).toBe("force_finish");
        expect(d.getNudgeCount()).toBe(2);
    });

    it("respects custom window size", () => {
        // Window of 2, threshold 2 → needs only 2 consecutive.
        const d = new DoomLoopDetector(2, 2);
        d.recordCall("a", "1");
        const r = d.recordCall("a", "1");
        expect(r.action).toBe("nudge");
    });

    it("a different call in between resets consecutive count", () => {
        const d = new DoomLoopDetector(5, 3);
        d.recordCall("search_messages", "x");
        d.recordCall("search_messages", "x");
        d.recordCall("note_add", "y"); // breaks streak
        d.recordCall("search_messages", "x");
        const r = d.recordCall("search_messages", "x");
        expect(r.action).toBe("ok");
    });
});

describe("ProgressTracker", () => {
    it("does not stall when progress tools are used", () => {
        const p = new ProgressTracker(3);
        expect(p.recordCall("search_messages", false).stalled).toBe(false);
        expect(p.recordCall("search_messages", false).stalled).toBe(false);
        expect(p.recordCall("note_add", false).stalled).toBe(false); // resets
        expect(p.recordCall("search_messages", false).stalled).toBe(false);
    });

    it("stalls after threshold non-progress calls", () => {
        const p = new ProgressTracker(3);
        p.recordCall("search_messages", false);
        p.recordCall("retrieve_messages", false);
        const r = p.recordCall("list_guild_structure", false);
        expect(r.stalled).toBe(true);
        expect(r.message).toContain("3 tool calls");
    });

    it("resets after stall nudge", () => {
        const p = new ProgressTracker(3);
        p.recordCall("a", false);
        p.recordCall("b", false);
        p.recordCall("c", false); // stall → resets
        // After reset, counter is back to 0.
        expect(p.recordCall("d", false).stalled).toBe(false);
        expect(p.recordCall("e", false).stalled).toBe(false);
    });

    it("evidence production counts as progress", () => {
        const p = new ProgressTracker(3);
        p.recordCall("search_messages", false);
        p.recordCall("search_messages", false);
        p.recordCall("search_messages", true); // evidence → reset
        expect(p.recordCall("x", false).stalled).toBe(false);
    });

    it("plan_update counts as progress", () => {
        const p = new ProgressTracker(2);
        p.recordCall("x", false);
        p.recordCall("plan_update", false); // progress
        expect(p.recordCall("y", false).stalled).toBe(false);
    });

    it("note_list counts as progress", () => {
        const p = new ProgressTracker(2);
        p.recordCall("x", false);
        p.recordCall("note_list", false); // progress
        expect(p.recordCall("y", false).stalled).toBe(false);
    });
});

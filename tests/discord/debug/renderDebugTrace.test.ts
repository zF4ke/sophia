import { describe, expect, it } from "vitest";
import { renderDebugTrace } from "@/discord/debug/renderDebugTrace";
import type { DebugTraceState } from "@/discord/debug/types";

function makeState(overrides: Partial<DebugTraceState> = {}): DebugTraceState {
    return {
        questionPreview: "Who decided that in the product channel?",
        status: "running",
        stage: "Running retrieve_messages",
        requesterLabel: "Talven",
        trigger: "mention",
        classificationMode: "discord_grounded",
        runtimeMode: "research",
        selectedCapabilities: ["retrieve_messages", "get_member_profile"],
        toolCallCount: 2,
        evidenceCount: 4,
        confidence: "best_effort",
        stopReason: "evidence_sufficient",
        stopDetail: "Scoped message evidence is available.",
        conversationThreadId: "g1:m-root:reply-chain",
        timeline: [
            {
                label: "tool_result",
                detail: "retrieve_messages: found strong evidence",
                tone: "success",
                timestamp: Date.now(),
            },
        ],
        startedAt: Date.now() - 1000,
        failureMessage: null,
        cumulativePromptTokens: 0,
        cumulativeCompletionTokens: 0,
        contextUsagePercent: null,
        notesSnapshot: [],
        planPreview: null,
        ...overrides,
    };
}

describe("renderDebugTrace", () => {
    it("renders overview with key fields and timeline", () => {
        const containers = renderDebugTrace(makeState());
        const rendered = JSON.stringify(
            containers.map((c) => c.toJSON())
        );

        expect(rendered).toContain("Sophia Debug Trace");
        expect(rendered).toContain("Talven");
        expect(rendered).toContain("mention");
        expect(rendered).toContain("discord_grounded");
        expect(rendered).toContain("retrieve_messages, get_member_profile");
        expect(rendered).toContain("2 calls");
        expect(rendered).toContain("4 items");
        expect(rendered).toContain("best_effort");
        expect(rendered).toContain("evidence_sufficient");
        expect(rendered).toContain("Timeline");
        expect(rendered).toContain("found strong evidence");
    });

    it("renders a failure message", () => {
        const containers = renderDebugTrace(makeState({
            status: "failed",
            stage: "Failed",
            failureMessage: "SQLITE_ERROR: timeout",
        }));
        const rendered = JSON.stringify(
            containers.map((c) => c.toJSON())
        );

        expect(rendered).toContain("FAILED");
        expect(rendered).toContain("SQLITE_ERROR: timeout");
    });

    it("renders conversation thread id", () => {
        const containers = renderDebugTrace(makeState({
            conversationThreadId: "g1:c1:channel",
        }));
        const rendered = JSON.stringify(
            containers.map((c) => c.toJSON())
        );

        expect(rendered).toContain("g1:c1:channel");
    });

    it("shows stop detail in overview", () => {
        const containers = renderDebugTrace(makeState({
            stopReason: "budget_exhausted",
            stopDetail: "Reached the latency budget.",
        }));
        const rendered = JSON.stringify(
            containers.map((c) => c.toJSON())
        );

        expect(rendered).toContain("budget_exhausted");
        expect(rendered).toContain("Reached the latency budget.");
    });

    it("handles long timelines by limiting visible entries", () => {
        const containers = renderDebugTrace(makeState({
            timeline: Array.from({ length: 30 }, (_, i) => ({
                label: `step-${i}`,
                detail: `detail ${i}`,
                tone: "info" as const,
                timestamp: Date.now() - i * 1000,
            })),
        }));
        const rendered = JSON.stringify(
            containers.map((c) => c.toJSON())
        );

        expect(rendered).toContain("10 earlier events");
    });

    it("handles empty timeline gracefully", () => {
        const containers = renderDebugTrace(makeState({ timeline: [] }));
        expect(containers.length).toBe(1); // just overview, no timeline
    });

    it("renders token usage and context percentage", () => {
        const containers = renderDebugTrace(makeState({
            cumulativePromptTokens: 12500,
            cumulativeCompletionTokens: 350,
            contextUsagePercent: 62.5,
        }));
        const rendered = JSON.stringify(
            containers.map((c) => c.toJSON())
        );

        expect(rendered).toContain("prompt");
        expect(rendered).toContain("completion");
        expect(rendered).toContain("62.5%");
        expect(rendered).toContain("Tokens");
        expect(rendered).toContain("Context");
    });

    it("renders notes scratchpad when notes exist", () => {
        const containers = renderDebugTrace(makeState({
            notesSnapshot: [
                { seq: 1, label: "phase-1", bodyPreview: "Found 200 messages about topic X", wordCount: 42 },
                { seq: 2, label: null, bodyPreview: "User spoke mostly in Portuguese", wordCount: 18 },
            ],
            planPreview: "Goal: Analyze linguistic evolution. Progress: 2/5 pages scanned.",
        }));
        const rendered = JSON.stringify(
            containers.map((c) => c.toJSON())
        );

        expect(rendered).toContain("Scratchpad");
        expect(rendered).toContain("2");
        expect(rendered).toContain("60");
        expect(rendered).toContain("phase-1");
        expect(rendered).toContain("Found 200 messages");
        expect(rendered).toContain("42w");
        expect(rendered).toContain("Plan");
        expect(rendered).toContain("Analyze linguistic evolution");
    });

    it("does not render scratchpad when no notes or plan exist", () => {
        const containers = renderDebugTrace(makeState({
            notesSnapshot: [],
            planPreview: null,
        }));
        const rendered = JSON.stringify(
            containers.map((c) => c.toJSON())
        );

        expect(rendered).not.toContain("Scratchpad");
    });
});

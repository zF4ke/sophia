import { describe, expect, it } from "vitest";
import type { RetrievalSummary } from "@/runtime/contracts";
import {
    buildForcedRetrieveArgs,
    buildRejectionMessage,
    createOrUpdate,
    incompleteFallback,
    isIncomplete,
    recordViolation,
    shouldForceContinuation,
    shouldRejectFinish,
    FORCE_CONTINUATION_AFTER_VIOLATIONS,
} from "@/runtime/corpusTaskState";

const DEFAULT_PAGE = 1000;

function retrieval(overrides: Partial<RetrievalSummary> = {}): RetrievalSummary {
    return {
        mode: "history",
        cacheHit: false,
        liveEscalated: false,
        searchedChannelIds: [],
        fetchedChannelIds: [],
        cacheEnriched: false,
        evidenceSufficient: false,
        strongResultCount: 0,
        weakResultCount: 0,
        historyMessageCount: 0,
        semanticMatchCount: 0,
        accumulatedUniqueCount: 0,
        sourceOrigin: "none",
        continuationAvailable: true,
        historyContinuationAvailable: true,
        historyCursorByChannel: {},
        semanticContinuationAvailable: false,
        semanticCursor: null,
        exhaustedChannelIds: [],
        historyExhausted: false,
        semanticExhausted: false,
        beforeTimestamp: null,
        afterTimestamp: null,
        activeChannelIds: ["C1"],
        ...overrides,
    };
}

describe("corpusTaskState", () => {
    it("does not activate when limit equals the default page size (a normal page)", () => {
        const state = createOrUpdate(
            undefined,
            retrieval({ accumulatedUniqueCount: 1000 }),
            { limit: 1000 },
            DEFAULT_PAGE,
        );
        expect(state).toBeUndefined();
    });

    it("activates when limit exceeds the default page size (explicit user total)", () => {
        const state = createOrUpdate(
            undefined,
            retrieval({ accumulatedUniqueCount: 1000 }),
            { limit: 20000, authorId: "A1", channelIds: ["C1"], query: "analise" },
            DEFAULT_PAGE,
        );
        expect(state).toBeDefined();
        expect(state!.requested).toBe(20000);
        expect(state!.collected).toBe(1000);
        expect(state!.targetAuthorId).toBe("A1");
        expect(state!.targetChannelIds).toEqual(["C1"]);
        expect((state!.originalArgs as { query?: string }).query).toBe("analise");
    });

    it("activates from the user's requested corpus size even when the model uses a tiny page", () => {
        const state = createOrUpdate(
            undefined,
            retrieval({ accumulatedUniqueCount: 100 }),
            { limit: 100, authorId: "A1", channelIds: ["C1"], query: "analise" },
            DEFAULT_PAGE,
            20000,
        );
        expect(state).toBeDefined();
        expect(state!.requested).toBe(20000);
        expect(state!.collected).toBe(100);
    });

    it("keeps tracking follow-up pages even when their limit is back at the default", () => {
        let state = createOrUpdate(undefined, retrieval({ accumulatedUniqueCount: 1000 }), { limit: 20000 }, DEFAULT_PAGE);
        state = createOrUpdate(state, retrieval({ accumulatedUniqueCount: 2000 }), { limit: 1000 }, DEFAULT_PAGE);
        state = createOrUpdate(state, retrieval({ accumulatedUniqueCount: 3000 }), { limit: 1000 }, DEFAULT_PAGE);
        expect(state!.collected).toBe(3000);
        expect(state!.requested).toBe(20000);
    });

    it("rejects finish while under target with continuation available", () => {
        const state = createOrUpdate(undefined, retrieval({ accumulatedUniqueCount: 3000 }), { limit: 20000 }, DEFAULT_PAGE);
        expect(isIncomplete(state)).toBe(true);
        expect(shouldRejectFinish(state)).toBe(true);
        const msg = buildRejectionMessage(state!);
        expect(msg).toContain("3000");
        expect(msg).toContain("20000");
    });

    it("allows finish once history is exhausted", () => {
        const state = createOrUpdate(
            undefined,
            retrieval({ accumulatedUniqueCount: 500, historyExhausted: true, continuationAvailable: false }),
            { limit: 20000 },
            DEFAULT_PAGE,
        );
        expect(isIncomplete(state)).toBe(false);
        expect(shouldRejectFinish(state)).toBe(false);
    });

    it("allows finish once collected meets requested", () => {
        const state = createOrUpdate(undefined, retrieval({ accumulatedUniqueCount: 20000 }), { limit: 20000 }, DEFAULT_PAGE);
        expect(isIncomplete(state)).toBe(false);
    });

    it("forces continuation after repeated violations across any guard", () => {
        const state = createOrUpdate(undefined, retrieval({ accumulatedUniqueCount: 3000 }), { limit: 20000 }, DEFAULT_PAGE)!;
        expect(shouldForceContinuation(state)).toBe(false);
        recordViolation(state);
        expect(shouldForceContinuation(state)).toBe(FORCE_CONTINUATION_AFTER_VIOLATIONS <= 1);
        recordViolation(state);
        expect(shouldForceContinuation(state)).toBe(true);
    });

    it("buildForcedRetrieveArgs preserves the full original retrieval shape plus cursor", () => {
        const state = createOrUpdate(
            undefined,
            retrieval({
                accumulatedUniqueCount: 3000,
                historyCursorByChannel: { C1: "msg123" },
            }),
            {
                limit: 20000,
                authorId: "A1",
                channelIds: ["C1"],
                query: "analise",
                mode: "history",
                order: "oldest",
                fromDate: "2024-01-01",
                beforeTimestamp: 1_700_000_000_000,
                afterTimestamp: 1_600_000_000_000,
                excludedMessageIds: ["m1", "m2"],
            },
            DEFAULT_PAGE,
        )!;
        const args = buildForcedRetrieveArgs(state);
        expect(args.query).toBe("analise");
        expect(args.authorId).toBe("A1");
        expect(args.channelIds).toEqual(["C1"]);
        expect(args.mode).toBe("history");
        expect(args.order).toBe("oldest");
        expect(args.fromDate).toBe("2024-01-01");
        expect(args.beforeTimestamp).toBe(1_700_000_000_000);
        expect(args.afterTimestamp).toBe(1_600_000_000_000);
        expect(args.excludedMessageIds).toEqual(["m1", "m2"]);
        expect(args.cursor).toBeDefined();
    });

    it("emits a deterministic bilingual incomplete-progress fallback", () => {
        const state = createOrUpdate(undefined, retrieval({ accumulatedUniqueCount: 3000 }), { limit: 20000 }, DEFAULT_PAGE)!;
        const out = incompleteFallback(state);
        expect(out).toContain("3000");
        expect(out).toContain("20000");
        expect(out).toMatch(/Tarefa não concluída/);
        expect(out).toMatch(/Task incomplete/);
    });
});

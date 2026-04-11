import { describe, expect, it } from "vitest";
import {
    extractDeterministicIntent,
    mergeIntent,
    normalize,
    parseModelIntent,
} from "@/runtime/intentExtraction";
import type { ActiveRetrievalSession, TurnIntent } from "@/runtime/contracts";

function createSession(overrides: Partial<ActiveRetrievalSession> = {}): ActiveRetrievalSession {
    return {
        mode: "history",
        channelIds: [],
        authorId: null,
        beforeTimestamp: null,
        afterTimestamp: null,
        historyCursorByChannel: {},
        semanticCursor: null,
        seenMessageIds: [],
        accumulatedUniqueCount: 0,
        exhaustedChannelIds: [],
        historyExhausted: false,
        semanticExhausted: false,
        continuationAvailable: false,
        ...overrides,
    };
}

describe("normalize", () => {
    it("strips accents and lowercases", () => {
        expect(normalize("Até Ontem")).toBe("ate ontem");
    });

    it("trims whitespace", () => {
        expect(normalize("  hello  ")).toBe("hello");
    });
});

describe("extractDeterministicIntent", () => {
    describe("continuation", () => {
        it("detects 'continue'", () => {
            const intent = extractDeterministicIntent("continue");
            expect(intent.continuation).toBe(true);
            expect(intent.source?.continuation).toBe("deterministic");
        });

        it("detects 'de novo'", () => {
            const intent = extractDeterministicIntent("de novo");
            expect(intent.continuation).toBe(true);
        });

        it("detects 'keep going'", () => {
            const intent = extractDeterministicIntent("keep going");
            expect(intent.continuation).toBe(true);
        });

        it("detects 'mais'", () => {
            const intent = extractDeterministicIntent("mais");
            expect(intent.continuation).toBe(true);
        });

        it("does not flag normal questions", () => {
            const intent = extractDeterministicIntent("what happened yesterday?");
            expect(intent.continuation).toBeUndefined();
        });
    });

    describe("retrieval mode", () => {
        it("returns 'semantic' for 'find references'", () => {
            const intent = extractDeterministicIntent("find references to auth");
            expect(intent.retrievalMode).toBe("semantic");
            expect(intent.source?.retrievalMode).toBe("deterministic");
        });

        it("returns 'semantic' for 'menções'", () => {
            const intent = extractDeterministicIntent("menções ao deploy");
            expect(intent.retrievalMode).toBe("semantic");
        });

        it("returns 'mixed' for 'all messages'", () => {
            const intent = extractDeterministicIntent("all messages from today");
            expect(intent.retrievalMode).toBe("mixed");
        });

        it("returns 'mixed' for 'todos'", () => {
            const intent = extractDeterministicIntent("todos os mensagens");
            expect(intent.retrievalMode).toBe("mixed");
        });

        it("returns undefined for normal questions", () => {
            const intent = extractDeterministicIntent("what did alice say");
            expect(intent.retrievalMode).toBeUndefined();
        });

        it("reuses active session mode", () => {
            const session = createSession({ mode: "semantic" });
            const intent = extractDeterministicIntent("what did alice say", session);
            expect(intent.retrievalMode).toBe("semantic");
            expect(intent.source?.retrievalMode).toBe("session");
        });

        it("prefers explicit deterministic mode over session mode", () => {
            const session = createSession({ mode: "semantic" });
            const intent = extractDeterministicIntent("all messages from atlas", session);
            expect(intent.retrievalMode).toBe("mixed");
            expect(intent.source?.retrievalMode).toBe("deterministic");
        });
    });

    describe("time bounds", () => {
        it("parses 'today'", () => {
            const intent = extractDeterministicIntent("what happened today");
            expect(intent.afterTimestamp).toBeDefined();
            expect(intent.beforeTimestamp).toBeUndefined();
            expect(intent.source?.timeBounds).toBe("deterministic");
        });

        it("parses 'hoje'", () => {
            const intent = extractDeterministicIntent("o que aconteceu hoje");
            expect(intent.afterTimestamp).toBeDefined();
        });

        it("parses 'yesterday' as a window", () => {
            const intent = extractDeterministicIntent("messages from yesterday");
            expect(intent.afterTimestamp).toBeDefined();
            expect(intent.beforeTimestamp).toBeDefined();
            const dayMs = 24 * 60 * 60 * 1000;
            expect(intent.beforeTimestamp! - intent.afterTimestamp!).toBe(dayMs);
        });

        it("parses 'before yesterday' as upper bound only", () => {
            const intent = extractDeterministicIntent("before yesterday");
            expect(intent.beforeTimestamp).toBeDefined();
            expect(intent.afterTimestamp).toBeUndefined();
        });

        it("parses 'until yesterday' as upper bound with startOfToday", () => {
            const intent = extractDeterministicIntent("until yesterday");
            expect(intent.beforeTimestamp).toBeDefined();
            expect(intent.afterTimestamp).toBeUndefined();
        });

        it("parses 'last week'", () => {
            const intent = extractDeterministicIntent("last week messages");
            expect(intent.afterTimestamp).toBeDefined();
            expect(intent.beforeTimestamp).toBeDefined();
            const dayMs = 24 * 60 * 60 * 1000;
            expect(intent.beforeTimestamp! - intent.afterTimestamp!).toBe(7 * dayMs);
        });

        it("parses 'ultima semana'", () => {
            const intent = extractDeterministicIntent("mensagens da ultima semana");
            expect(intent.afterTimestamp).toBeDefined();
            expect(intent.beforeTimestamp).toBeDefined();
        });

        it("parses explicit date with 'before'", () => {
            const intent = extractDeterministicIntent("before 2024-01-15");
            expect(intent.beforeTimestamp).toBe(
                Date.parse("2024-01-15T00:00:00") + 24 * 60 * 60 * 1000
            );
        });

        it("parses explicit date with 'after'", () => {
            const intent = extractDeterministicIntent("after 2024-06-01");
            expect(intent.afterTimestamp).toBe(Date.parse("2024-06-01T00:00:00"));
        });

        it("returns no time bounds for generic questions", () => {
            const intent = extractDeterministicIntent("who is alice");
            expect(intent.beforeTimestamp).toBeUndefined();
            expect(intent.afterTimestamp).toBeUndefined();
        });
    });
});

describe("parseModelIntent", () => {
    it("extracts continuation from model response", () => {
        const intent = parseModelIntent({
            intent: { continuation: true, retrievalMode: null, beforeDate: null, afterDate: null },
        });
        expect(intent.continuation).toBe(true);
        expect(intent.source?.continuation).toBe("model");
    });

    it("extracts retrieval mode", () => {
        const intent = parseModelIntent({
            intent: { continuation: null, retrievalMode: "semantic", beforeDate: null, afterDate: null },
        });
        expect(intent.retrievalMode).toBe("semantic");
        expect(intent.source?.retrievalMode).toBe("model");
    });

    it("converts ISO date strings to timestamps", () => {
        const intent = parseModelIntent({
            intent: { continuation: null, retrievalMode: null, beforeDate: "2024-03-15", afterDate: "2024-03-01" },
        });
        expect(intent.beforeTimestamp).toBe(
            Date.parse("2024-03-15T00:00:00") + 24 * 60 * 60 * 1000
        );
        expect(intent.afterTimestamp).toBe(Date.parse("2024-03-01T00:00:00"));
        expect(intent.source?.timeBounds).toBe("model");
    });

    it("returns empty for missing intent block", () => {
        const intent = parseModelIntent({ mode: "research" });
        expect(intent).toEqual({});
    });

    it("returns empty for null input", () => {
        const intent = parseModelIntent(null);
        expect(intent).toEqual({});
    });

    it("ignores invalid retrieval mode values", () => {
        const intent = parseModelIntent({
            intent: { retrievalMode: "invalid_mode" },
        });
        expect(intent.retrievalMode).toBeUndefined();
    });
});

describe("mergeIntent", () => {
    it("deterministic wins over model for continuation", () => {
        const result = mergeIntent(
            { continuation: true, source: { continuation: "deterministic" } as any },
            { continuation: false, source: { continuation: "model" } as any }
        );
        expect(result.continuation).toBe(true);
        expect(result.source.continuation).toBe("deterministic");
    });

    it("model fills gaps when deterministic has no signal", () => {
        const result = mergeIntent(
            {},
            { continuation: true, source: { continuation: "model" } as any }
        );
        expect(result.continuation).toBe(true);
        expect(result.source.continuation).toBe("model");
    });

    it("defaults to false/null/none when neither layer has signal", () => {
        const result = mergeIntent({}, {});
        expect(result.continuation).toBe(false);
        expect(result.retrievalMode).toBeNull();
        expect(result.beforeTimestamp).toBeNull();
        expect(result.afterTimestamp).toBeNull();
        expect(result.source.continuation).toBe("deterministic");
        expect(result.source.retrievalMode).toBe("none");
        expect(result.source.timeBounds).toBe("none");
    });

    it("deterministic time bounds win over model time bounds", () => {
        const result = mergeIntent(
            {
                beforeTimestamp: 1000,
                afterTimestamp: 500,
                source: { timeBounds: "deterministic" } as any,
            },
            {
                beforeTimestamp: 2000,
                afterTimestamp: 1500,
                source: { timeBounds: "model" } as any,
            }
        );
        expect(result.beforeTimestamp).toBe(1000);
        expect(result.afterTimestamp).toBe(500);
        expect(result.source.timeBounds).toBe("deterministic");
    });

    it("model time bounds used when deterministic has none", () => {
        const result = mergeIntent(
            {},
            {
                beforeTimestamp: 2000,
                source: { timeBounds: "model" } as any,
            }
        );
        expect(result.beforeTimestamp).toBe(2000);
        expect(result.afterTimestamp).toBeNull();
        expect(result.source.timeBounds).toBe("model");
    });

    it("fills missing deterministic time fields from model", () => {
        const result = mergeIntent(
            {
                beforeTimestamp: 1000,
                source: { timeBounds: "deterministic" } as any,
            },
            {
                afterTimestamp: 500,
                source: { timeBounds: "model" } as any,
            }
        );

        expect(result.beforeTimestamp).toBe(1000);
        expect(result.afterTimestamp).toBe(500);
        expect(result.source.timeBounds).toBe("deterministic");
    });

    it("session retrieval mode preserved through deterministic layer", () => {
        const result = mergeIntent(
            { retrievalMode: "semantic", source: { retrievalMode: "session" } as any },
            { retrievalMode: "history", source: { retrievalMode: "model" } as any }
        );
        expect(result.retrievalMode).toBe("semantic");
        expect(result.source.retrievalMode).toBe("session");
    });
});

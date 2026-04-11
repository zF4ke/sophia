import { describe, expect, it } from "vitest";
import { mergeActiveRetrievalSession } from "@/runtime/Runtime";
import type { ActiveRetrievalSession } from "@/runtime/contracts";

function createSession(
    overrides: Partial<ActiveRetrievalSession> = {}
): ActiveRetrievalSession {
    return {
        mode: "history",
        channelIds: ["c-atlas"],
        authorId: null,
        beforeTimestamp: null,
        afterTimestamp: null,
        historyCursorByChannel: { "c-atlas": "105" },
        semanticCursor: null,
        seenMessageIds: ["105", "104"],
        accumulatedUniqueCount: 2,
        exhaustedChannelIds: [],
        historyExhausted: false,
        semanticExhausted: true,
        continuationAvailable: true,
        ...overrides,
    };
}

describe("retrieval session merging", () => {
    it("keeps accumulatedUniqueCount monotonic and unions seen ids across continuation pages", () => {
        const merged = mergeActiveRetrievalSession(
            createSession(),
            createSession({
                historyCursorByChannel: { "c-atlas": "103" },
                seenMessageIds: ["103", "102"],
                accumulatedUniqueCount: 4,
            })
        );

        expect(merged).toEqual(
            expect.objectContaining({
                historyCursorByChannel: { "c-atlas": "103" },
                seenMessageIds: ["105", "104", "103", "102"],
                accumulatedUniqueCount: 4,
            })
        );
    });

    it("resets instead of merging when the retrieval scope changes", () => {
        const next = createSession({
            channelIds: ["c-traveller"],
            historyCursorByChannel: { "c-traveller": "301" },
            seenMessageIds: ["301"],
            accumulatedUniqueCount: 1,
        });

        expect(mergeActiveRetrievalSession(createSession(), next)).toEqual(next);
    });
});

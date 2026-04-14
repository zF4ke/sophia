import { describe, expect, it } from "vitest";
import {
    answerConfidenceForInsufficient,
    classify,
    countEvidence,
    formatChannelContext,
} from "@/runtime/planning";
import type { EvidenceItem, ChannelContextMessage } from "@/runtime/contracts";

function makeEvidence(overrides: Partial<EvidenceItem> = {}): EvidenceItem {
    return {
        tool: "retrieve_messages",
        summary: "test",
        content: "test content",
        evidenceRole: "message_evidence",
        strength: "strong",
        sourceOrigin: "cache",
        ...overrides,
    };
}

describe("runtime planning", () => {
    describe("classify", () => {
        it("returns discord_grounded for research mode", () => {
            const result = classify("research");
            expect(result.mode).toBe("discord_grounded");
        });

        it("returns direct_answer for conversation mode", () => {
            const result = classify("conversation");
            expect(result.mode).toBe("direct_answer");
        });
    });

    describe("countEvidence", () => {
        it("counts message evidence by strength", () => {
            const counts = countEvidence({
                evidence: [
                    makeEvidence({ strength: "strong" }),
                    makeEvidence({ strength: "weak" }),
                    makeEvidence({ strength: "strong" }),
                ],
            });
            expect(counts.messageEvidenceCount).toBe(3);
            expect(counts.strongMessageEvidenceCount).toBe(2);
            expect(counts.weakMessageEvidenceCount).toBe(1);
            expect(counts.liveEvidenceCount).toBe(0);
        });

        it("counts live evidence separately", () => {
            const counts = countEvidence({
                evidence: [
                    makeEvidence({ evidenceRole: "live_evidence", strength: "strong" }),
                    makeEvidence({ evidenceRole: "discovery_only", strength: "weak" }),
                ],
            });
            expect(counts.messageEvidenceCount).toBe(0);
            expect(counts.liveEvidenceCount).toBe(2);
        });

        it("does not count history evidence toward answer confidence", () => {
            const counts = countEvidence({
                evidence: [makeEvidence({ evidenceRole: "history_evidence", strength: "strong" })],
            });

            expect(counts.messageEvidenceCount).toBe(0);
            expect(counts.strongMessageEvidenceCount).toBe(0);
            expect(counts.weakMessageEvidenceCount).toBe(0);
        });

        it("returns zeros for empty evidence", () => {
            const counts = countEvidence({ evidence: [] });
            expect(counts.messageEvidenceCount).toBe(0);
            expect(counts.liveEvidenceCount).toBe(0);
        });
    });

    describe("answerConfidenceForInsufficient", () => {
        it("returns best_effort when strong evidence exists", () => {
            const mode = answerConfidenceForInsufficient({
                evidence: [makeEvidence({ strength: "strong" })],
            });
            expect(mode).toBe("best_effort");
        });

        it("returns best_effort when at least 2 message evidence items exist", () => {
            const mode = answerConfidenceForInsufficient({
                evidence: [
                    makeEvidence({ strength: "weak" }),
                    makeEvidence({ strength: "weak" }),
                ],
            });
            expect(mode).toBe("best_effort");
        });

        it("returns insufficient when evidence is too thin", () => {
            const mode = answerConfidenceForInsufficient({
                evidence: [makeEvidence({ strength: "weak" })],
            });
            expect(mode).toBe("insufficient");
        });

        it("returns insufficient when only history evidence exists", () => {
            const mode = answerConfidenceForInsufficient({
                evidence: [makeEvidence({ evidenceRole: "history_evidence", strength: "strong" })],
            });
            expect(mode).toBe("insufficient");
        });

        it("returns insufficient for empty evidence", () => {
            const mode = answerConfidenceForInsufficient({ evidence: [] });
            expect(mode).toBe("insufficient");
        });
    });

    describe("formatChannelContext", () => {
        it("formats messages as author: content lines", () => {
            const ctx: ChannelContextMessage[] = [
                { authorName: "Alice", content: "hello", createdTimestamp: 1 },
                { authorName: "Bob", content: "hi", createdTimestamp: 2 },
            ];
            expect(formatChannelContext(ctx)).toBe("Alice: hello\nBob: hi");
        });

        it("returns placeholder for empty context", () => {
            expect(formatChannelContext([])).toBe("No recent channel messages available.");
        });
    });
});

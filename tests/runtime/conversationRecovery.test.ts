import { describe, expect, it } from "vitest";
import {
    buildConversationalRecovery,
    buildDirectConversationFallback,
    sanitizeConversationalAnswer,
} from "@/runtime/conversationRecovery";

describe("conversation recovery", () => {
    it("keeps greeting fallback conversational and away from dead-end banned strings", () => {
        const answer = buildDirectConversationFallback("oi sophia");

        expect(answer).toContain("Oi");
        expect(answer).not.toBe("I couldn't ground that in Discord evidence.");
        expect(answer).not.toBe("I don't have enough Discord evidence to answer that yet.");
    });

    it("sanitizes dead-end answers before they can be shown to the user", () => {
        expect(
            sanitizeConversationalAnswer("I couldn't ground that in Discord evidence.")
        ).toBe("");
        expect(
            sanitizeConversationalAnswer("I don't have enough Discord evidence to answer that yet.")
        ).toBe("");
        expect(sanitizeConversationalAnswer("Let me think this through with you.")).toBe(
            "Let me think this through with you."
        );
    });

    it("turns weak follow-up recovery into a continuous conversational handoff", () => {
        const answer = buildConversationalRecovery({
            question: "what did she mean by that?",
            confidence: "insufficient",
            evidence: [],
            replyContext: {
                messageId: "m1",
                authorId: "u1",
                authorName: "alice",
                authorDisplayName: "Alice",
                content: "We should probably hold the launch for one more day.",
                jumpLink: null,
            },
            priorTurns: [
                {
                    requestId: "r1",
                    question: "who was talking about the launch?",
                    answer: "Alice was the one discussing the launch delay.",
                    classificationMode: "discord_grounded",
                    runtimeMode: "research",
                    stopReason: "evidence_sufficient",
                    confidence: "best_effort",
                    createdTimestamp: Date.now(),
                },
            ],
        });

        expect(answer).toContain("replying to:");
        expect(answer).toContain("Earlier you asked");
        expect(answer).toContain("Can you give me more details?");
        expect(answer).not.toContain("I couldn't ground that in Discord evidence.");
        expect(answer).not.toContain("Based on what I found");
        expect(answer).not.toContain("Recent context: the last turn");
    });

    it("produces natural evidence hints without raw data labels", () => {
        const answer = buildConversationalRecovery({
            question: "o que o bob disse?",
            confidence: "best_effort",
            evidence: [
                {
                    tool: "retrieve_messages",
                    summary: "Found messages",
                    content: "Vamos adiar o lancamento",
                    evidenceRole: "message_evidence",
                    strength: "strong",
                    sourceOrigin: "cache",
                    authorName: "Bob",
                    channelName: "general",
                },
            ],
        });

        expect(answer).toContain("Bob");
        expect(answer).toContain("mencionou");
        expect(answer).not.toContain("Based on what I found");
        expect(answer).not.toContain("from cached Discord history");
        expect(answer).not.toContain("retrieve_messages");
    });

    it("mentions research time exhaustion instead of pretending nothing exists", () => {
        const answer = buildConversationalRecovery({
            question: "me explica o que tem no atlas",
            confidence: "insufficient",
            evidence: [],
            stopReason: "budget_exhausted",
        });

        expect(answer).toContain("Fiquei sem tempo de pesquisa");
        expect(answer).not.toContain("Ainda nao encontrei o que preciso");
    });

    it("does not hardcode disambiguation — lets model handle ambiguous members naturally", () => {
        const answer = buildConversationalRecovery({
            question: "qual glonos e o verdadeiro?",
            confidence: "insufficient",
            evidence: [
                {
                    tool: "list_members",
                    summary: "2 members listed",
                    content: "Glonos (@subjectless)",
                    evidenceRole: "live_evidence",
                    strength: "metadata",
                    sourceOrigin: "none",
                    authorId: "u-subjectless",
                    authorName: "Glonos",
                },
                {
                    tool: "list_members",
                    summary: "2 members listed",
                    content: "Glonos (@glonos)",
                    evidenceRole: "live_evidence",
                    strength: "metadata",
                    sourceOrigin: "none",
                    authorId: "u-glonos",
                    authorName: "Glonos",
                },
            ],
            stopReason: "no_useful_next_step",
        });

        expect(answer).toContain("Glonos");
        expect(answer).toContain("mentioned");
        expect(answer).not.toContain("Encontrei mais de um perfil com esse nome");
        expect(answer).not.toContain("qual deles");
    });
});

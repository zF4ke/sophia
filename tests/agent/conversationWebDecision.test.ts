import { describe, expect, it } from "vitest";
import { decideConversationWebMode } from "@/agent/orchestration/conversationWebDecision";

describe("decideConversationWebMode", () => {
    it("keeps web off for casual conversational chat", () => {
        const decision = decideConversationWebMode({
            question: "eae blz?",
            classification: {
                mode: "direct_answer",
                reason: "general chat",
            },
            conversationSurface: "talk",
            conversationWebMode: "auto",
        });

        expect(decision.webMode).toBe("off");
    });

    it("enables web for conversational current external questions", () => {
        const decision = decideConversationWebMode({
            question: "quais foram as noticias de IA hoje?",
            classification: {
                mode: "direct_answer",
                reason: "current external question",
            },
            conversationSurface: "mention",
            conversationWebMode: "auto",
        });

        expect(decision.webMode).toBe("auto");
    });

    it("keeps web off when Discord grounding already answered the request", () => {
        const decision = decideConversationWebMode({
            question: "quem falou isso no servidor?",
            classification: {
                mode: "discord_grounded",
                reason: "Discord evidence required",
            },
            conversationSurface: "reply",
            conversationWebMode: "auto",
            groundedAnswerMode: "confident",
        });

        expect(decision.webMode).toBe("off");
    });

    it("enables web as a selective fallback for unresolved external grounded questions", () => {
        const decision = decideConversationWebMode({
            question: "esse anuncio de IA de hoje tem mais detalhes oficiais?",
            classification: {
                mode: "discord_grounded",
                reason: "mixed Discord + external context",
            },
            conversationSurface: "talk",
            conversationWebMode: "auto",
            groundedAnswerMode: "insufficient",
        });

        expect(decision.webMode).toBe("auto");
    });
});

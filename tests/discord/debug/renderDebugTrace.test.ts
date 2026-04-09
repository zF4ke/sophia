import { describe, expect, it } from "vitest";
import { renderDebugTrace } from "@/discord/debug/renderDebugTrace";

describe("renderDebugTrace", () => {
    it("renders a running grounded trace", () => {
        const json = renderDebugTrace({
            questionPreview: "Quem decidiu isso no canal de produto?",
            status: "running",
            stage: "Usando search_messages",
            mode: "Com grounding do Discord",
            controllerDecision: {
                source: "ai",
                questionIntent: "channel_or_topic_search",
                routeIntent: "channel_target",
                nextAction: "search_messages",
                targetText: "produto",
                needsMessageEvidence: true,
                answerConfidence: "best_effort",
                confidence: 0.82,
                reason: "Canal parece ser o alvo.",
            },
            toolNames: ["search_messages", "read_message_thread"],
            groundingSummary: {
                messageEvidenceCount: 4,
                liveEvidenceCount: 0,
                sufficient: true,
            },
            groundingDecisionMode: "judge",
            groundedAnswerMode: "best_effort",
            contextCacheStatus: "seeded",
            recentEvents: ["Executando search_messages", "Modo escolhido: Com grounding do Discord"],
            startedAt: Date.now() - 1_000,
        }).toJSON();

        expect(JSON.stringify(json)).toContain("Debug da Sophia");
        expect(JSON.stringify(json)).toContain("Usando search_messages");
        expect(JSON.stringify(json)).toContain("search_messages, read_message_thread");
        expect(JSON.stringify(json)).toContain("channel_or_topic_search");
        expect(JSON.stringify(json)).toContain("Base útil");
        expect(JSON.stringify(json)).toContain("mensagens 4");
        expect(JSON.stringify(json)).toContain("Cache de contexto");
        expect(JSON.stringify(json)).toContain("juiz");
    });

    it("renders a failed trace", () => {
        const json = renderDebugTrace({
            questionPreview: "teste",
            status: "failed",
            stage: "Falhou",
            mode: "Resposta direta",
            controllerDecision: null,
            toolNames: [],
            groundingSummary: {
                messageEvidenceCount: 0,
                liveEvidenceCount: 0,
                sufficient: false,
            },
            groundingDecisionMode: "heuristic",
            groundedAnswerMode: null,
            contextCacheStatus: "none",
            recentEvents: ["Erro: timeout"],
            startedAt: Date.now() - 1_000,
        }).toJSON();

        expect(JSON.stringify(json)).toContain("Falhou");
        expect(JSON.stringify(json)).toContain("Erro: timeout");
        expect(JSON.stringify(json)).not.toContain("Grounding");
    });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway } from "@/ai/ModelGateway";
import { decideGroundingSufficiency } from "@/agent/orchestration/evidenceJudge";

describe("decideGroundingSufficiency", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("uses the model judge for ambiguous middle-ground evidence", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            sufficient: true,
            reason: "The evidence directly lists the requested songs.",
            missingInformation: null,
        });

        const result = await decideGroundingSufficiency({
            question: "pega as musicas do canal scart",
            grounding: {
                summary: {
                    messageEvidenceCount: 2,
                    liveEvidenceCount: 0,
                    sufficient: false,
                },
                evidence: "1. [scart] F4zke: A realidade se transforma",
            },
            toolRuns: [
                {
                    tool: "search_messages",
                    summary: "Found 2 relevant message chunks.",
                    data: [
                        {
                            totalScore: 0.31,
                        },
                    ],
                } as any,
            ],
            routeDecision: {
                source: "ai",
                intent: "channel_target",
                targetText: "scart",
                confidence: 0.9,
                reason: "channel",
            },
        });

        expect(result).toEqual({
            sufficient: true,
            mode: "judge",
            answerMode: "confident",
            reason: "The evidence directly lists the requested songs.",
            missingInformation: null,
        });
    });

    it("skips the model for obviously insufficient evidence", async () => {
        const generateJsonSpy = vi.spyOn(ModelGateway, "generateJson");

        const result = await decideGroundingSufficiency({
            question: "o que aconteceu?",
            grounding: {
                summary: {
                    messageEvidenceCount: 0,
                    liveEvidenceCount: 0,
                    sufficient: false,
                },
                evidence: "",
            },
            toolRuns: [],
            routeDecision: {
                source: "deterministic",
                intent: "broad_search",
                targetText: null,
                confidence: 1,
                reason: "broad",
            },
        });

        expect(generateJsonSpy).not.toHaveBeenCalled();
        expect(result).toEqual({
            sufficient: false,
            mode: "heuristic",
            answerMode: "insufficient",
            reason: "No direct evidence is available yet.",
            missingInformation: "More Discord evidence is needed.",
        });
    });

    it("skips the model for obviously sufficient server-context evidence", async () => {
        const generateJsonSpy = vi.spyOn(ModelGateway, "generateJson");

        const result = await decideGroundingSufficiency({
            question: "que servidor é esse?",
            grounding: {
                summary: {
                    messageEvidenceCount: 0,
                    liveEvidenceCount: 1,
                    sufficient: true,
                },
                evidence: "Server name: Oz Synthesis",
            },
            toolRuns: [
                {
                    tool: "get_guild_context",
                    summary: "Oz Synthesis: 17 membros e 68 canais.",
                    data: {
                        id: "g1",
                        name: "Oz Synthesis",
                        memberCount: 17,
                        channelCount: 68,
                    },
                },
            ],
            routeDecision: {
                source: "deterministic",
                intent: "server_context",
                targetText: null,
                confidence: 1,
                reason: "server",
            },
        });

        expect(generateJsonSpy).not.toHaveBeenCalled();
        expect(result.mode).toBe("heuristic");
        expect(result.sufficient).toBe(true);
    });
});

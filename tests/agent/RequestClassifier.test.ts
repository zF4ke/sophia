import { describe, expect, it, vi } from "vitest";
import { RequestClassifier } from "@/agent/RequestClassifier";
import { ModelGateway } from "@/ai/ModelGateway";

describe("RequestClassifier", () => {
    it("classifies general questions as direct answers", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            mode: "direct_answer",
            reason: "general",
        });

        const result = await RequestClassifier.classify("Qual a capital de Portugal?");
        expect(result.mode).toBe("direct_answer");
    });

    it("classifies discord history questions as discord grounded", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            mode: "discord_grounded",
            reason: "discord-specific",
        });

        const result = await RequestClassifier.classify(
            "Quem disse isso no canal geral ontem?"
        );
        expect(result.mode).toBe("discord_grounded");
    });

    it("treats perguntas sobre este servidor as discord grounded heuristically", () => {
        const result = RequestClassifier.classifyHeuristically(
            "Pode me dizer que servidor é esse?"
        );

        expect(result.mode).toBe("discord_grounded");
    });
});

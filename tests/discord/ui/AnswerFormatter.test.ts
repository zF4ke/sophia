import { describe, expect, it } from "vitest";
import { AnswerFormatter } from "@/discord/ui/formatters/AnswerFormatter";

describe("AnswerFormatter", () => {
    it("returns only the answer body even when citations exist", () => {
        const formatted = AnswerFormatter.format("Resposta pronta.", [
            {
                label: "scart · F4zke",
                jumpLink: "https://discord.com/channels/g1/c1/m1",
            },
        ]);

        expect(formatted).toBe("Resposta pronta.");
        expect(formatted).not.toContain("discord.com/channels");
    });
});

import { describe, expect, it } from "vitest";
import { runArtifactScript, SCRIPT_LIMITS } from "@/discord/artifacts/ArtifactScript";

const base = {
    state: { score: 0 } as Record<string, unknown>,
    user: { id: "u1", username: "F4zke" },
    values: [] as string[],
    customId: "action:roll",
    cardId: "m1",
};

describe("artifact script container integration", () => {
    it("mutates persisted state and replies", async () => {
        const result = await runArtifactScript({
            ...base,
            code: "state.score = (state.score || 0) + 5; reply(`score ${state.score}`);",
        });

        expect(result.error).toBeNull();
        expect(result.state.score).toBe(5);
        expect(result.reply).toBe("score 5");
        expect(result.sends).toHaveLength(0);
    });

    it("exposes user, values and customId", async () => {
        const result = await runArtifactScript({
            ...base,
            customId: "action:pick",
            values: ["pizza", "sushi"],
            code: "reply(`${user.username} escolheu ${values.join(', ')} em ${customId}`);",
        });

        expect(result.reply).toBe("F4zke escolheu pizza, sushi em action:pick");
    });

    it("queues sends and caps them", async () => {
        const result = await runArtifactScript({
            ...base,
            code: "send('123456789', 'primeira'); send('123456789', 'segunda'); send('123456789', 'terceira'); send('123456789', 'quarta');",
        });

        expect(result.sends).toHaveLength(3);
        expect(result.sends[2].content).toBe("terceira");
    });

    it("supports render helpers", async () => {
        const result = await runArtifactScript({
            ...base,
            code: "setTitle('Novo título'); setAccent(0xff0000); setSpoiler(false); setSection(2);",
        });

        expect(result.title).toBe("Novo título");
        expect(result.accentColor).toBe(0xff0000);
        expect(result.spoiler).toBe(false);
        expect(result.section).toBe(2);
    });

    it("rejects rejected-code and returns the error to the caller", async () => {
        const result = await runArtifactScript({
            ...base,
            code: "thisDoesNotExist()",
        });

        expect(result.error).toContain("ReferenceError");
        expect(result.state).toEqual(base.state);
        expect(result.reply).toBeNull();
    });

    it("blocks sandbox escapes", async () => {
        // require/process/fetch must be undefined inside the sandbox.
        const noRequire = await runArtifactScript({ ...base, code: "reply(`require=${typeof require} process=${typeof process} globalThis.fetch=${typeof globalThis.fetch}`);" });
        expect(noRequire.reply).toBe("require=undefined process=undefined globalThis.fetch=undefined");
        expect(noRequire.error).toBeNull();
    });

    it("enforces the code size limit", async () => {
        const result = await runArtifactScript({
            ...base,
            code: "x".repeat(SCRIPT_LIMITS.maxCodeChars + 1),
        });

        expect(result.error).toContain("exceeds");
    });

    it("protects oversized state from clobbering the card", async () => {
        const result = await runArtifactScript({
            ...base,
            code: "state.blob = 'x'.repeat(5000);",
        });

        expect(result.error).toBeNull();
        expect(result.state).toEqual({ score: 0 });
    });
});

import { describe, expect, it } from "vitest";
import { DebugSession } from "@/discord/debug/DebugSession";

function createFakeMessage(id: string) {
    return {
        id,
        edit: async () => undefined,
    } as any;
}

describe("DebugSession", () => {
    it("reuses collapsed section preferences for new sessions", async () => {
        const first = new DebugSession(createFakeMessage("m1"), "primeira pergunta");
        await first.toggleSection("timeline");

        const second = new DebugSession(createFakeMessage("m2"), "segunda pergunta");
        const rendered = JSON.stringify(
            second.buildComponents().map((component: { toJSON(): unknown }) => component.toJSON())
        );

        expect(rendered).not.toContain("Sophia Debug · Timeline");
    });
});

import { describe, expect, it } from "vitest";
import { DebugSession } from "@/discord/debug/DebugSession";

function createFakeMessage(id: string) {
    return {
        id,
        edit: async () => undefined,
    } as any;
}

describe("DebugSession", () => {
    it("tracks tool calls and capabilities through the session", async () => {
        const session = new DebugSession(createFakeMessage("m1"), "test question");
        await session.setToolResult("retrieve_messages", "found evidence", 5);
        await session.setToolResult("get_member_profile", "resolved member");

        const rendered = JSON.stringify(
            session.buildComponents().map((component: { toJSON(): unknown }) => component.toJSON())
        );

        expect(rendered).toContain("retrieve_messages");
        expect(rendered).toContain("get_member_profile");
        expect(rendered).toContain("2 calls");
    });

    it("records failure message on error", async () => {
        const session = new DebugSession(createFakeMessage("m2"), "failing question");
        await session.finishError(new Error("timeout"));

        const rendered = JSON.stringify(
            session.buildComponents().map((component: { toJSON(): unknown }) => component.toJSON())
        );

        expect(rendered).toContain("FAILED");
        expect(rendered).toContain("timeout");
    });
});

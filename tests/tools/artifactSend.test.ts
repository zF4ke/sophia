import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { ArtifactStore } from "@/discord/artifacts/ArtifactStore";

function makeContext(guild: unknown = undefined) {
    return {
        guild: guild as never,
        question: "send an artifact",
        currentChannelId: "c1",
    };
}

describe("artifact_send tool", () => {
    afterEach(() => vi.restoreAllMocks());
    it("preserves the sent message ID and reports a failed state save without a duplicate send", async () => {
        const send = vi.fn().mockResolvedValue({ id: "sent-card", url: "https://discord.com/channels/g1/c1/sent-card" });
        const channel = { id: "c1", isTextBased: () => true, send };
        vi.spyOn(ArtifactStore, "record").mockRejectedValue(new Error("disk full"));
        const result = await CapabilityRegistry.get("artifact_send").run(makeContext({ id: "g1", channels: { cache: new Map([["c1", channel]]) } }), { title: "Report", sections: [{ body: "Result" }] });
        expect(send).toHaveBeenCalledOnce();
        expect(result.data).toMatchObject({ messageId: "sent-card", channelId: "c1" });
        expect(result.errorMessage).toContain("editable state could not be saved");
    });
    beforeEach(() => {
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
    });

    it("rejects invalid specs before touching Discord", async () => {
        const result = await CapabilityRegistry.get("artifact_send").run(makeContext({ id: "g1" }), {
            title: "",
            sections: [{ body: "x" }],
        } as never);

        expect(result.errorMessage).toBeTruthy();
        expect(result.errorMessage).toMatch(/title/i);
    });

    it("requires a guild context", async () => {
        const result = await CapabilityRegistry.get("artifact_send").run(makeContext(), {
            title: "Ok",
            sections: [{ body: "x" }],
        } as never);

        expect(result.errorMessage).toBe("No guild context.");
    });

    it("reports a missing target channel without sending", async () => {
        const result = await CapabilityRegistry.get("artifact_send").run(makeContext({ id: "g1", channels: { cache: new Map() } }), {
            title: "Ok",
            sections: [{ body: "x" }],
            channel_id: "missing",
        } as never);

        expect(result.errorMessage).toMatch(/not found or not text-based/i);
    });

    it("accepts sections passed as a JSON string instead of an array", async () => {
        const result = await CapabilityRegistry.get("artifact_send").run(makeContext({ id: "g1", channels: { cache: new Map() } }), {
            title: "Ok",
            sections: JSON.stringify([{ body: "x" }]),
            channel_id: "missing",
        } as never);

        // Passes validation and Zod, failing only at the (missing) channel lookup.
        expect(result.errorMessage).toMatch(/not found or not text-based/i);
        expect(result.errorMessage).not.toMatch(/invalid input/i);
    });
});

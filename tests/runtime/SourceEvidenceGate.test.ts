import { afterEach, expect, it, vi } from "vitest";
import { z } from "zod";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { ToolExecutor } from "@/runtime/ToolExecutor";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { readableToolRecords } from "@/runtime/sourceEvidence";
import { ExecutionControl } from "@/runtime/ExecutionControl";

afterEach(() => vi.restoreAllMocks());
it("does not dispatch a write decided before a pending evidence update", async () => {
    const capability = CapabilityRegistry.get("send_message");
    const run = vi.fn();
    vi.spyOn(CapabilityRegistry, "get").mockReturnValue({ ...capability, run });
    const execution = new ExecutionControl("owner", "channel");
    const release = execution.register();
    try {
        execution.watchSources(["edited"]);
        ExecutionControl.invalidateSource("edited", { kind: "edited" });
        const result = await ToolExecutor.execute("send_message", { channel_id: "channel", content: "Old decision" }, {
            guild: { id: "guild" } as never, actorId: "owner", currentChannelId: "channel", question: "Send it", execution, authorize: async () => "allow",
        });
        expect(result.record.blocked).toBe(true);
        expect(result.resultPayload).toContain("source changed");
        expect(run).not.toHaveBeenCalled();
        expect(execution.signal.aborted).toBe(false);
    } finally { release(); }
});
it("withholds message payloads after permission revocation or source deletion", async () => {
    const capability = CapabilityRegistry.get("retrieve_messages");
    const jumpLink = "https://discord.com/channels/guild/source/evidence-gate";
    const run = vi.fn().mockResolvedValue({ tool: "retrieve_messages", summary: "SECRET_SOURCE", data: { historyMessages: [{ messageId: "evidence-gate", channelId: "source", authorId: "author", content: "SECRET_SOURCE", jumpLink, createdTimestamp: 1 }], semanticMatches: [] } });
    vi.spyOn(CapabilityRegistry, "get").mockReturnValue({ ...capability, outputSchema: z.any(), run });
    const has = vi.fn().mockReturnValue(false);
    const guild = { id: "guild", members: { fetch: vi.fn().mockResolvedValue({}) }, channels: { fetch: vi.fn().mockResolvedValue({ isTextBased: () => true, permissionsFor: () => ({ has }) }) } };
    const context = { guild: guild as never, actorId: "owner", question: "Read", authorize: async () => "allow" as const };
    const denied = await ToolExecutor.execute("retrieve_messages", { query: "*" }, context);
    expect(denied.record.blocked).toBe(true);
    expect(JSON.stringify(denied)).not.toContain("SECRET_SOURCE");
    has.mockReturnValue(true);
    const allowed = await ToolExecutor.execute("retrieve_messages", { query: "*" }, context);
    expect(allowed.resultPayload).toContain("SECRET_SOURCE");
    has.mockReturnValue(false);
    expect(JSON.stringify(await readableToolRecords([allowed.record], guild as never, "owner"))).not.toContain("SECRET_SOURCE");
    has.mockReturnValue(true);
    await knowledgeStore.invalidateSource(jumpLink);
    const deleted = await ToolExecutor.execute("retrieve_messages", { query: "*" }, context);
    expect(deleted.record.blocked).toBe(true);
    expect(JSON.stringify(deleted)).not.toContain("SECRET_SOURCE");
});

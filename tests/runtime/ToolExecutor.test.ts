import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { ToolExecutor } from "@/runtime/ToolExecutor";
import { ExecutionControl } from "@/runtime/ExecutionControl";
import { SettingsService } from "@/app/SettingsService";

describe("ToolExecutor", () => {
    it("denies protected targets inside the executor even with Auto or caller approval", async () => {
        vi.restoreAllMocks();
        const capability = CapabilityRegistry.get("delete_channel");
        const run = vi.fn();
        vi.spyOn(CapabilityRegistry, "get").mockReturnValue({ ...capability, run });
        const settings = SettingsService.load();
        vi.spyOn(SettingsService, "load").mockReturnValue({ ...settings, protectedChannelIds: ["protected"] });
        const result = await ToolExecutor.execute("delete_channel", { channel_id: "protected" }, { guild: null, question: "delete", authorize: async () => "allow" }, { approved: true });
        expect(run).not.toHaveBeenCalled();
        expect(result.record.learned).toContain("protected");
    });
    it("serializes competing mutations and rechecks a queued request after revocation", async () => {
        vi.restoreAllMocks();
        let release!: () => void;
        let started!: () => void;
        const waiting = new Promise<void>(resolve => { release = resolve; });
        const entered = new Promise<void>(resolve => { started = resolve; });
        const capability = CapabilityRegistry.get("send_message");
        const run = vi.fn(async () => { started(); await waiting; return { tool: "send_message" as const, summary: "sent", data: {} }; });
        vi.spyOn(CapabilityRegistry, "get").mockReturnValue({ ...capability, run });
        let allowed = true;
        const context = { guild: null, currentChannelId: "serialized-channel", question: "send", authorize: async () => allowed ? "allow" as const : "deny" as const };
        const first = ToolExecutor.execute("send_message", { channel_id: "serialized-channel", content: "first" }, context);
        await entered;
        const second = ToolExecutor.execute("send_message", { channel_id: "serialized-channel", content: "second" }, context);
        allowed = false;
        release();
        await first;
        const result = await second;
        expect(run).toHaveBeenCalledTimes(1);
        expect(result.record.blocked).toBe(true);
        expect(result.uncertainAction).not.toBe(true);
    });
    it("skips approved work if steering arrives during the final authorization check", async () => {
        vi.restoreAllMocks();
        const execution = new ExecutionControl("owner", "channel");
        const capability = CapabilityRegistry.get("send_message");
        const run = vi.fn();
        vi.spyOn(CapabilityRegistry, "get").mockReturnValue({ ...capability, run });
        const result = await ToolExecutor.execute("send_message", { channel_id: "c1", content: "old instruction" },
            { guild: null, question: "send", execution, authorize: async () => {
                execution.steer("Change the destination.");
                return "ask";
            } }, { approved: true, steeringRevision: 0 });
        expect(run).not.toHaveBeenCalled();
        expect(result.record.blocked).toBe(true);
        expect(result.record.learned).toContain("correction superseded");
    });
    it("does not execute when auto permission changed to ask before dispatch", async () => {
        vi.restoreAllMocks();
        const capability = CapabilityRegistry.get("send_message");
        const run = vi.fn();
        vi.spyOn(CapabilityRegistry, "get").mockReturnValue({ ...capability, run });
        const result = await ToolExecutor.execute("send_message", { channel_id: "c1", content: "hi", approved: true },
            { guild: null, question: "send", authorize: async () => "ask" });
        expect(run).not.toHaveBeenCalled();
        expect(result.record.learned).toContain("requires approval");
    });
    it("rechecks revoked authority before running a queued capability", async () => {
        vi.restoreAllMocks();
        const capability = CapabilityRegistry.get("send_message");
        const run = vi.fn();
        vi.spyOn(CapabilityRegistry, "get").mockReturnValue({ ...capability, run });
        const authorize = vi.fn().mockResolvedValue("deny");
        const result = await ToolExecutor.execute("send_message", { channel_id: "c1", content: "hi", actorId: "operator" },
            { guild: null, question: "send", actorId: "u1", authorize });
        expect(authorize).toHaveBeenCalledWith("write", "send_message");
        expect(run).not.toHaveBeenCalled();
        expect(result.record.blocked).toBe(true);
    });
    it("rejects invalid arguments before invoking a capability", async () => {
        const run = vi.fn();
        vi.spyOn(CapabilityRegistry, "get").mockReturnValue({
            id: "evaluate_math",
            kind: "tool",
            description: "test",
            inputSchema: z.object({ expression: z.string().min(1) }),
            outputSchema: z.any(),
            sideEffectLevel: "none",
            authRequirements: [],
            costClass: "cheap",
            latencyClass: "fast",
            evidenceRole: "discovery_only",
            preconditions: [],
            postconditions: [],
            run,
        });

        const result = await ToolExecutor.execute(
            "evaluate_math",
            {},
            { guild: null, question: "calculate" },
        );

        expect(run).not.toHaveBeenCalled();
        expect(result.record.blocked).toBe(true);
        expect(result.record.learned).toContain("Invalid arguments");
    });

    it("times out a stuck read-only capability", async () => {
        vi.spyOn(CapabilityRegistry, "get").mockReturnValue({
            id: "evaluate_math",
            kind: "tool",
            description: "test",
            inputSchema: z.object({}),
            outputSchema: z.any(),
            sideEffectLevel: "none",
            authRequirements: [],
            costClass: "cheap",
            latencyClass: "fast",
            evidenceRole: "discovery_only",
            preconditions: [],
            postconditions: [],
            run: vi.fn(() => new Promise<never>(() => {})),
        });

        const result = await ToolExecutor.execute(
            "evaluate_math",
            {},
            { guild: null, question: "calculate" },
            { timeoutMs: 10 },
        );

        expect(result.record.blocked).toBe(true);
        expect(result.record.learned).toContain("Timed out after 10ms");
    });

    it("validates the capability data without treating the result envelope as data", async () => {
        vi.spyOn(CapabilityRegistry, "get").mockReturnValue({
            id: "evaluate_math",
            kind: "tool",
            description: "test",
            inputSchema: z.object({}),
            outputSchema: z.object({ result: z.number() }),
            sideEffectLevel: "none",
            authRequirements: [],
            costClass: "cheap",
            latencyClass: "fast",
            evidenceRole: "discovery_only",
            preconditions: [],
            postconditions: [],
            run: vi.fn().mockResolvedValue({
                tool: "evaluate_math",
                summary: "result",
                data: { result: 4 },
            }),
        });

        const result = await ToolExecutor.execute(
            "evaluate_math",
            {},
            { guild: null, question: "calculate" },
        );

        expect(result.record.blocked).not.toBe(true);
        expect(result.record.output.data).toEqual({ result: 4 });
    });
});

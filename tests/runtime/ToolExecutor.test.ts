import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { ToolExecutor } from "@/runtime/ToolExecutor";

describe("ToolExecutor", () => {
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

import { describe, expect, it } from "vitest";
import { startLongTaskTool } from "@/tools/startLongTask";

describe("startLongTask tool definition", () => {
    it("has the correct name", () => {
        expect(startLongTaskTool.name).toBe("start_long_task");
    });

    it("has effect 'read' so it never triggers the approval gate", () => {
        expect(startLongTaskTool.catalog.effect).toBe("read");
    });

    it("has sideEffectLevel 'none'", () => {
        expect(startLongTaskTool.capability.sideEffectLevel).toBe("none");
    });

    it("has a schema with required 'reason' parameter", () => {
        const schema = startLongTaskTool.schema.parameters;
        expect(schema).toBeDefined();
        expect((schema as any).required).toContain("reason");
    });

    it("no longer exposes estimate parameters (operator-configured caps)", () => {
        const props = (startLongTaskTool.schema.parameters as any).properties;
        expect(props.estimated_tool_calls).toBeUndefined();
        expect(props.estimated_seconds).toBeUndefined();
    });

    it("extracts no evidence", () => {
        const result = startLongTaskTool.strategy.extractEvidence({
            tool: "start_long_task",
            summary: "ok",
            data: {},
            errorMessage: null,
        } as any);
        expect(result).toEqual([]);
    });
});

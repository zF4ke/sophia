import { describe, expect, it, vi } from "vitest";
import { PromptRegistry } from "@/runtime/PromptRegistry";

describe("prompt interpolation", () => {
    it("does not interpret placeholders inside replacement values", () => {
        vi.spyOn(PromptRegistry, "load").mockReturnValue("Name: {{name}}; Role: {{role}}");
        expect(PromptRegistry.render("test", { name: "{{role}}", role: "reader" }))
            .toBe("Name: {{role}}; Role: reader");
    });
});

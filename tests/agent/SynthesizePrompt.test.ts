import { readFileSync } from "fs";
import { join } from "path";
import { describe, expect, it } from "vitest";

describe("synthesize_answer prompt", () => {
    it("forbids Evidence headings", () => {
        const prompt = readFileSync(
            join(process.cwd(), "resources", "prompts", "tasks", "synthesize_answer.md"),
            "utf8"
        );

        expect(prompt).toContain('Do not prepend headings like "Evidence:"');
    });
});

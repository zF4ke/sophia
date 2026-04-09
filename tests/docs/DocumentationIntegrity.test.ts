import fs from "fs";
import path from "path";
import { describe, expect, it } from "vitest";
import { DISCORD_TOOL_NAMES } from "@/shared/discordTools";
import { RUNTIME_PROMPT_IDS } from "@/shared/promptCatalog";

const projectRoot = process.cwd();
const importantDocs = [
    "docs/architecture.md",
    "docs/agent-loop.md",
    "docs/memory-indexing.md",
    "docs/prompt-catalog.md",
    "docs/commands-and-admin.md",
    "docs/cleanup-migration.md",
];

describe("documentation integrity", () => {
    it("keeps AGENTS.md references in sync with the important docs hub", () => {
        const agents = fs.readFileSync(path.join(projectRoot, "AGENTS.md"), "utf8");

        for (const docPath of importantDocs) {
            expect(fs.existsSync(path.join(projectRoot, docPath))).toBe(true);
            expect(agents).toContain(docPath);
        }
    });

    it("keeps runtime prompt files present for the catalog", () => {
        for (const promptId of RUNTIME_PROMPT_IDS) {
            const promptPath = path.join(
                projectRoot,
                "resources",
                "prompts",
                `${promptId}.md`
            );
            expect(fs.existsSync(promptPath)).toBe(true);
        }
    });

    it("keeps tool names synchronized across docs and the planning prompt", () => {
        const agents = fs.readFileSync(path.join(projectRoot, "AGENTS.md"), "utf8");
        const promptCatalog = fs.readFileSync(
            path.join(projectRoot, "docs/prompt-catalog.md"),
            "utf8"
        );
        const plannerPrompt = fs.readFileSync(
            path.join(projectRoot, "resources/prompts/tasks/plan_discord_search.md"),
            "utf8"
        );

        for (const toolName of DISCORD_TOOL_NAMES) {
            expect(agents).toContain(toolName);
            expect(promptCatalog).toContain(toolName);
            expect(plannerPrompt).toContain(toolName);
        }
    });
});

import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";

describe("plan_discord_search prompt", () => {
    it("documents live metadata and discovery-only behavior", () => {
        const prompt = readFileSync(
            join(process.cwd(), "resources", "prompts", "tasks", "plan_discord_search.md"),
            "utf-8"
        );

        expect(prompt).toContain("current-server facts");
        expect(prompt).toContain("discovery only");
        expect(prompt).toContain("finish");
        expect(prompt).toContain("list_members(filters, limit, offset, sort)");
        expect(prompt).toContain("crawl_channel_messages(channelId, limit, queryHint)");
    });
});

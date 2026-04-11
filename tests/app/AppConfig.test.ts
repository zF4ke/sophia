import { afterEach, describe, expect, it, vi } from "vitest";

describe("AppConfig", () => {
    afterEach(() => {
        delete process.env.DISCORD_TOKEN;
        delete process.env.CLIENT_TOKEN;
        delete process.env.OPENROUTER_API_KEY;
        delete process.env.OPENAI_API_KEY;
        delete process.env.RUNTIME_MAX_RESEARCH_PASSES;
        vi.resetModules();
    });

    it("defaults runtime research passes to four", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.runtime.maxResearchPasses).toBe(4);
    });

    it("keeps runtime research passes configurable through env", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";
        process.env.RUNTIME_MAX_RESEARCH_PASSES = "6";

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.runtime.maxResearchPasses).toBe(6);
    });
});

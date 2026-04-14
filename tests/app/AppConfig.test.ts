import { afterEach, describe, expect, it, vi } from "vitest";

describe("AppConfig", () => {
    afterEach(() => {
        delete process.env.DISCORD_TOKEN;
        delete process.env.OPENROUTER_API_KEY;
        vi.resetModules();
    });

    it("defaults runtime max tool calls to six", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.runtime.maxToolCalls).toBe(6);
    });

    it("reads runtime config from settings", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.runtime.interactiveCrawlLimit).toBe(250);
        expect(config.runtime.escalationFetchLimit).toBe(150);
        expect(config.runtime.retrievalHistoryLimit).toBe(50);
        expect(config.runtime.retrievalContextWindow).toBe(15);
    });
});

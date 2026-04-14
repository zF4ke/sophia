import { afterEach, describe, expect, it, vi } from "vitest";

describe("AppConfig", () => {
    afterEach(() => {
        delete process.env.DISCORD_TOKEN;
        delete process.env.OPENROUTER_API_KEY;
        delete process.env.MODEL_PROFILE;
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

    it("uses selected settings profile instead of MODEL_PROFILE env override", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";
        process.env.MODEL_PROFILE = "fast";

        const { SettingsService } = await import("@/app/SettingsService");
        const defaults = SettingsService.getDefaults();
        vi.spyOn(SettingsService, "load").mockReturnValue({
            ...defaults,
            modelProfile: "smarter",
        });

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.modelProfileName).toBe("smarter");
        expect(config.modelProfile.chatModel).toBe("minimax/minimax-m2.7");
    });
});

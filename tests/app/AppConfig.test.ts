import { afterEach, describe, expect, it, vi } from "vitest";

describe("AppConfig", () => {
    afterEach(() => {
        delete process.env.DISCORD_TOKEN;
        delete process.env.OPENROUTER_API_KEY;
        delete process.env.MODEL_PROFILE;
        vi.resetModules();
    });

    it("reads runtime max tool calls from settings", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.runtime.maxToolCalls).toBeGreaterThanOrEqual(2);
    });

    it("reads runtime config from settings", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";

        const { SettingsService } = await import("@/app/SettingsService");
        const settings = SettingsService.load();
        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.runtime.escalationFetchLimit).toBe(settings.runtime.escalationFetchLimit);
        expect(config.runtime.retrievalHistoryLimit).toBe(settings.runtime.retrievalHistoryLimit);
        expect(config.runtime.retrievalContextWindow).toBe(settings.runtime.retrievalContextWindow);
    });

    it("uses selected settings profile instead of MODEL_PROFILE env override", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";
        process.env.MODEL_PROFILE = "fast";

        const { SettingsService } = await import("@/app/SettingsService");
        const defaults = SettingsService.getDefaults();
        const updateSpy = vi.spyOn(SettingsService, "update").mockImplementation((patch) => ({
            ...defaults,
            ...patch,
            runtime: defaults.runtime,
        }));
        vi.spyOn(SettingsService, "load").mockReturnValue({
            ...defaults,
            modelProfile: "smarter",
        });

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.modelProfileName).toBe("minimax27");
        expect(config.modelProfile.chatModel).toBe("minimax/minimax-m2.7");
        expect(updateSpy).toHaveBeenCalledWith({ modelProfile: "minimax27" });
    });

    it("falls back from stale saved profile names", async () => {
        process.env.DISCORD_TOKEN = "discord-token";
        process.env.OPENROUTER_API_KEY = "openrouter-key";

        const { SettingsService } = await import("@/app/SettingsService");
        const defaults = SettingsService.getDefaults();
        const updateSpy = vi.spyOn(SettingsService, "update").mockImplementation((patch) => ({
            ...defaults,
            ...patch,
            runtime: defaults.runtime,
        }));
        vi.spyOn(SettingsService, "load").mockReturnValue({
            ...defaults,
            modelProfile: "cheap",
        });

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.modelProfileName).toBe("free-elephant");
        expect(config.modelProfile.chatModel).toBe("openrouter/elephant-alpha");
        expect(updateSpy).toHaveBeenCalledWith({ modelProfile: "free-elephant" });
    });
});

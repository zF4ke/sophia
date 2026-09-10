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
            modelProfile: "ling30flash",
        });

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.modelProfileName).toBe("ling30flash");
        expect(config.modelProfile.chatModel).toBe("inclusionai/ling-3.0-flash");
        expect(updateSpy).not.toHaveBeenCalled();
    });

    it("falls back from invalid saved profile names", async () => {
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
            modelProfile: "does-not-exist",
        });

        const { getAppConfig } = await import("@/app/AppConfig");
        const config = getAppConfig();

        expect(config.modelProfileName).toBe(defaults.modelProfile);
        expect(updateSpy).toHaveBeenCalledWith({ modelProfile: defaults.modelProfile });
    });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import { SettingsService } from "@/app/SettingsService";

describe("DebugModeService", () => {
    beforeEach(() => {
        DebugModeService.resetForTests();
        vi.restoreAllMocks();
    });

    it("defaults to disabled when no stored state exists", () => {
        vi.spyOn(SettingsService, "load").mockReturnValue({ ...SettingsService.getDefaults(), debug: false });

        expect(DebugModeService.isEnabled()).toBe(false);
    });

    it("persists enable and disable toggles", () => {
        const updateSpy = vi.spyOn(SettingsService, "update").mockImplementation((patch) => {
            return { ...SettingsService.getDefaults(), ...patch } as any;
        });

        DebugModeService.setEnabled(true);
        expect(DebugModeService.isEnabled()).toBe(true);
        expect(updateSpy).toHaveBeenLastCalledWith({ debug: true });

        DebugModeService.setEnabled(false);
        expect(DebugModeService.isEnabled()).toBe(false);
        expect(updateSpy).toHaveBeenLastCalledWith({ debug: false });
    });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import { DebugStateStore } from "@/discord/debug/DebugStateStore";

describe("DebugModeService", () => {
    beforeEach(() => {
        DebugModeService.resetForTests();
        vi.restoreAllMocks();
    });

    it("defaults to disabled when no stored state exists", () => {
        vi.spyOn(DebugStateStore, "load").mockReturnValue({ enabled: false });

        expect(DebugModeService.isEnabled()).toBe(false);
    });

    it("persists enable and disable toggles", () => {
        const saveSpy = vi.spyOn(DebugStateStore, "save").mockImplementation(() => undefined);

        DebugModeService.setEnabled(true);
        expect(DebugModeService.isEnabled()).toBe(true);
        expect(saveSpy).toHaveBeenLastCalledWith({ enabled: true });

        DebugModeService.setEnabled(false);
        expect(DebugModeService.isEnabled()).toBe(false);
        expect(saveSpy).toHaveBeenLastCalledWith({ enabled: false });
    });
});

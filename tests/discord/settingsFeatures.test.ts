import { afterEach, expect, it, vi } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { SecurityService } from "@/security/SecurityService";
import { handleSettingsPanelInteraction } from "@/discord/commands/system/settings/settingsInteractions";
import { buildSettingsPanel } from "@/discord/commands/system/settings/settings.command";

afterEach(() => vi.restoreAllMocks());
it("toggles services without changing grants and denies non-operators", async () => {
    vi.spyOn(SecurityService, "initialize").mockResolvedValue();
    const admin = vi.spyOn(SecurityService, "isAdmin").mockReturnValue(true);
    const original = SettingsService.load();
    SettingsService.update({ access: { ...original.access, directMessages: true, users: [{ userId: "owner", mode: "ask", level: "write" }] } });
    const before = SettingsService.load();
    const input = { customId: "settings:sandbox:toggle", user: { id: "operator" }, update: vi.fn().mockResolvedValue(undefined), reply: vi.fn().mockResolvedValue(undefined) };
    await handleSettingsPanelInteraction(input as never);
    expect(SettingsService.load().sandbox.enabled).toBe(!before.sandbox.enabled);
    expect(SettingsService.load().access).toEqual(before.access);
    input.customId = "settings:scheduling:toggle";
    await handleSettingsPanelInteraction(input as never);
    expect(SettingsService.load().scheduling.enabled).toBe(!before.scheduling.enabled);
    input.customId = "settings:dreaming:toggle";
    await handleSettingsPanelInteraction(input as never);
    expect(SettingsService.load().memory.dreamingEnabled).toBe(!before.memory.dreamingEnabled);
    admin.mockReturnValue(false);
    const protectedState = SettingsService.load();
    await handleSettingsPanelInteraction(input as never);
    expect(SettingsService.load()).toEqual(protectedState);
    expect(input.reply).toHaveBeenCalledWith(expect.objectContaining({ flags: 64 }));
    // Discord builder validation checks the actual serialized component tree.
    const panel = buildSettingsPanel(protectedState, "features");
    expect(JSON.stringify(panel.components.map(component => component.toJSON()))).toContain("settings:scheduling:toggle");
});

it("uses unique custom IDs throughout every serialized settings tab", () => {
    for (const tab of ["model", "runtime", "compaction", "longTask", "voice", "features"] as const) {
        const panel = buildSettingsPanel(SettingsService.load(), tab);
        const ids: string[] = [];
        function visit(value: unknown): void {
            if (!value || typeof value !== "object") return;
            if ("custom_id" in value) ids.push(String(value.custom_id));
            for (const child of Object.values(value)) visit(child);
        }
        visit(panel.components.map(component => component.toJSON()));
        expect(ids.length, tab).toBeGreaterThan(0);
        expect(new Set(ids).size, tab).toBe(ids.length);
    }
});

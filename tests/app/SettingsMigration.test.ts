import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { SettingsService } from "@/app/SettingsService";

describe("v5 execution settings migration", () => {
    it("resets only the chosen category and preserves availability and durable storage paths", () => {
        fs.mkdirSync(AppPaths.storageRoot, { recursive: true });
        fs.writeFileSync(path.join(AppPaths.storageRoot, "settings.json"), JSON.stringify(SettingsService.getDefaults()));
        SettingsService.invalidateCache();
        const configured = SettingsService.update({ guildAllowlist: ["guild"], protectedChannelIds: ["protected"],
            access: { ...SettingsService.getDefaults().access, users: [{ userId: "owner", mode: "ask", level: "write" }] }, voice: "casual",
            runtime: { ...SettingsService.load().runtime, operationalDbPath: "storage/custom.sqlite", toolCallLimit: 700 } });
        const reset = SettingsService.reset("runtime");
        expect(reset.runtime.toolCallLimit).toBe(0);
        expect(reset.runtime.operationalDbPath).toBe("storage/custom.sqlite");
        expect(reset.access).toEqual(configured.access);
        expect(reset.guildAllowlist).toEqual(["guild"]);
        expect(reset.protectedChannelIds).toEqual(["protected"]);
        expect(reset.voice).toBe("casual");
        expect(() => SettingsService.update({ compaction: { ...reset.compaction, triggerFraction: 2 } })).toThrow();
        expect(SettingsService.load()).toEqual(reset);
    });
    it("migrates voice without changing access grants or retaining persona overrides", () => {
        const legacy: any = { ...SettingsService.getDefaults(), personality: "mixed" };
        delete legacy.voice;
        fs.mkdirSync(AppPaths.storageRoot, { recursive: true });
        fs.writeFileSync(path.join(AppPaths.storageRoot, "settings.json"), JSON.stringify(legacy));
        SettingsService.invalidateCache();
        const migrated = SettingsService.load();
        expect(migrated.voice).toBe("casual");
        expect(migrated.access).toEqual(legacy.access);
        const saved = JSON.parse(fs.readFileSync(path.join(AppPaths.storageRoot, "settings.json"), "utf8"));
        expect(saved.personality).toBeUndefined();
        expect(saved.voice).toBe("casual");
    });
    it("does not replace invalid settings with permissive defaults", () => {
        fs.mkdirSync(AppPaths.storageRoot, { recursive: true });
        fs.writeFileSync(path.join(AppPaths.storageRoot, "settings.json"), "{broken");
        SettingsService.invalidateCache();
        expect(() => SettingsService.load()).toThrow("Unable to load settings");
        expect(fs.readFileSync(path.join(AppPaths.storageRoot, "settings.json"), "utf8")).toBe("{broken");
    });
    it("retires old budgets without losing other settings", () => {
        const settings = SettingsService.getDefaults();
        const legacy = { ...settings, runtime: { ...settings.runtime, maxToolCalls: 25,
            longTask: { ...settings.runtime.longTask, maxToolCalls: 200 } } } as any;
        delete legacy.runtime.toolCallLimit;
        legacy.runtime.retrievalHistoryLimit = 750;
        fs.mkdirSync(AppPaths.storageRoot, { recursive: true });
        fs.writeFileSync(path.join(AppPaths.storageRoot, "settings.json"), JSON.stringify(legacy));
        SettingsService.invalidateCache();
        const migrated = SettingsService.load();
        expect(migrated.runtime.toolCallLimit).toBe(0);
        expect(migrated.runtime.retrievalHistoryLimit).toBe(750);
        const saved = JSON.parse(fs.readFileSync(path.join(AppPaths.storageRoot, "settings.json"), "utf8"));
        expect(saved.runtime.maxToolCalls).toBeUndefined();
        expect(saved.runtime.longTask.maxToolCalls).toBeUndefined();
        SettingsService.update({ runtime: { ...migrated.runtime, toolCallLimit: 500 } });
        SettingsService.invalidateCache();
        expect(SettingsService.load().runtime.toolCallLimit).toBe(500);
    });
});

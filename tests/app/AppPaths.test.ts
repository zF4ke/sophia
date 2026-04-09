import fs from "fs";
import { describe, expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { FileSystemService } from "@/platform/storage/FileSystemService";

describe("AppPaths", () => {
    it("resolves runtime assets from resources", () => {
        expect(AppPaths.promptsRoot.endsWith("resources\\prompts")).toBe(true);
        expect(AppPaths.modelProfilesPath.endsWith("resources\\models\\model-profiles.json")).toBe(
            true
        );
        expect(fs.existsSync(AppPaths.promptsRoot)).toBe(true);
        expect(fs.existsSync(AppPaths.modelProfilesPath)).toBe(true);
    });

    it("resolves mutable runtime state from storage", () => {
        expect(AppPaths.storageRoot.endsWith("storage")).toBe(true);
        expect(FileSystemService.getBaseStorageDir()).toBe(AppPaths.storageRoot);
    });
});

import fs from "fs";
import { describe, expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { FileSystemService } from "@/shared/storage/FileSystemService";

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
        // Under tests, SOPHIA_STORAGE_ROOT redirects to an isolated dir
        // (see tests/setup/testStorage.ts); that path still terminates in "storage".
        expect(AppPaths.storageRoot.endsWith("storage")).toBe(true);
        expect(FileSystemService.getBaseStorageDir()).toBe(AppPaths.storageRoot);
    });
});


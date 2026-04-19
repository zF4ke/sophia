import fs from "fs";
import path from "path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { RuntimeStorageService } from "@/runtime/storage/RuntimeStorageService";

function uniqueRuntimeDir(): string {
    return path.join(
        process.cwd(),
        "storage",
        "test-runtime-storage-status",
        `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    );
}

describe("RuntimeStorageService", () => {
    let runtimeDir = "";
    let operationalDbPath = "";
    let checkpointDbPath = "";

    beforeEach(() => {
        runtimeDir = uniqueRuntimeDir();
        operationalDbPath = path.join(runtimeDir, "operational.sqlite");
        checkpointDbPath = path.join(runtimeDir, "checkpoints.sqlite");
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        fs.mkdirSync(runtimeDir, { recursive: true });
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath,
                checkpointDbPath,
            },
        });
    });

    afterEach(() => {
        try {
            fs.rmSync(runtimeDir, { recursive: true, force: true });
        } catch {
            // ignore cleanup errors in temp test dirs
        }
    });

    it("reports sqlite family sizes for status surfaces", () => {
        fs.writeFileSync(operationalDbPath, Buffer.alloc(1024));
        fs.writeFileSync(`${operationalDbPath}-wal`, Buffer.alloc(512));
        fs.writeFileSync(`${checkpointDbPath}-shm`, Buffer.alloc(256));

        const status = RuntimeStorageService.getStatus();

        expect(status.operationalDbSizeBytes).toBe(1536);
        expect(status.checkpointDbSizeBytes).toBe(256);
    });
});

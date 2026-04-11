import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { getAppConfig } from "@/app/AppConfig";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import { CheckpointStore } from "@/runtime/storage/CheckpointStore";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import {
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_VERSION_FILE,
    OPERATIONAL_SCHEMA_VERSION,
} from "@/runtime/storage/schema";

async function wait(ms: number): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, ms));
}

async function deletePathWithRetries(target: string): Promise<void> {
    for (let attempt = 0; attempt < 10; attempt += 1) {
        if (!fs.existsSync(target)) {
            return;
        }

        try {
            fs.rmSync(target, { force: true });
            return;
        } catch (error) {
            const code = error && typeof error === "object" && "code" in error ? String((error as NodeJS.ErrnoException).code) : "";
            if ((code === "EBUSY" || code === "EPERM") && attempt < 4) {
                await wait(50 * (attempt + 1));
                continue;
            }
            throw error;
        }
    }
}

async function deleteSqliteFamily(filePath: string): Promise<void> {
    for (const suffix of ["", "-wal", "-shm"]) {
        await deletePathWithRetries(`${filePath}${suffix}`);
    }
}

function clearDirectoryContents(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
        return;
    }

    for (const entry of fs.readdirSync(dirPath)) {
        const target = path.join(dirPath, entry);
        try {
            fs.rmSync(target, { recursive: true, force: true });
        } catch (error) {
            const code = error && typeof error === "object" && "code" in error ? String((error as NodeJS.ErrnoException).code) : "";
            if (code !== "EBUSY" && code !== "EPERM") {
                throw error;
            }
        }
    }
}

export class RuntimeStorageService {
    public static getStatus() {
        const config = getAppConfig();
        const runtimeDir = path.dirname(config.runtime.operationalDbPath);
        const checkpointVersionPath = path.join(runtimeDir, CHECKPOINT_VERSION_FILE);

        return {
            operationalDbPath: config.runtime.operationalDbPath,
            checkpointDbPath: config.runtime.checkpointDbPath,
            operationalSchemaVersion: OPERATIONAL_SCHEMA_VERSION,
            checkpointSchemaVersion: CHECKPOINT_SCHEMA_VERSION,
            checkpointVersionPath,
            runtimeDir,
            logsDir: path.join(AppPaths.storageRoot, "logs"),
        };
    }

    public static async resetAllRuntimeData(): Promise<void> {
        const status = this.getStatus();
        await OperationalStore.reset();
        await CheckpointStore.reset();

        await deleteSqliteFamily(status.operationalDbPath);
        await deleteSqliteFamily(status.checkpointDbPath);

        if (fs.existsSync(status.checkpointVersionPath)) {
            fs.rmSync(status.checkpointVersionPath, { force: true });
        }

        clearDirectoryContents(status.logsDir);
        clearDirectoryContents(path.join(AppPaths.storageRoot, "runtime"));
        clearDirectoryContents(path.join(AppPaths.storageRoot, "memory"));
        clearDirectoryContents(path.join(AppPaths.storageRoot, "test-memory"));

        FileSystemService.ensureDirectoryExists(path.join(AppPaths.storageRoot, "runtime"));
        FileSystemService.ensureDirectoryExists(path.join(AppPaths.storageRoot, "logs"));
    }
}




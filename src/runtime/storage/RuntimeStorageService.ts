import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { getAppConfig } from "@/app/AppConfig";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { OPERATIONAL_SCHEMA_VERSION } from "@/runtime/storage/schema";

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

function getPathSize(filePath: string): number {
    if (!fs.existsSync(filePath)) {
        return 0;
    }
    return fs.statSync(filePath).size;
}

function getSqliteFamilySize(filePath: string): number {
    return ["", "-wal", "-shm"].reduce(
        (total, suffix) => total + getPathSize(`${filePath}${suffix}`),
        0,
    );
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

        return {
            operationalDbPath: config.runtime.operationalDbPath,
            checkpointDbPath: config.runtime.checkpointDbPath,
            operationalDbSizeBytes: getSqliteFamilySize(config.runtime.operationalDbPath),
            checkpointDbSizeBytes: getSqliteFamilySize(config.runtime.checkpointDbPath),
            operationalSchemaVersion: OPERATIONAL_SCHEMA_VERSION,
            runtimeDir,
            logsDir: path.join(AppPaths.storageRoot, "logs"),
        };
    }

    public static async resetAllRuntimeData(): Promise<void> {
        const status = this.getStatus();
        await OperationalStore.reset();

        await deleteSqliteFamily(status.operationalDbPath);
        await deleteSqliteFamily(status.checkpointDbPath);

        clearDirectoryContents(status.logsDir);
        clearDirectoryContents(path.join(AppPaths.storageRoot, "runtime"));
        clearDirectoryContents(path.join(AppPaths.storageRoot, "memory"));
        clearDirectoryContents(path.join(AppPaths.storageRoot, "test-memory"));

        FileSystemService.ensureDirectoryExists(path.join(AppPaths.storageRoot, "runtime"));
        FileSystemService.ensureDirectoryExists(path.join(AppPaths.storageRoot, "logs"));
    }
}



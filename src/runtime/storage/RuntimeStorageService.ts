import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { getAppConfig } from "@/app/AppConfig";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { OPERATIONAL_SCHEMA_VERSION } from "@/runtime/storage/schema";

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
            operationalDbSizeBytes: getSqliteFamilySize(config.runtime.operationalDbPath),
            durableStores: ["tasks.sqlite", "knowledge.sqlite", "products.sqlite"].map(name => ({ name, sizeBytes: getSqliteFamilySize(path.join(AppPaths.storageRoot, name)) })),
            operationalSchemaVersion: OPERATIONAL_SCHEMA_VERSION,
            runtimeDir,
            logsDir: path.join(AppPaths.storageRoot, "logs"),
        };
    }

    public static async resetAllRuntimeData(): Promise<void> {
        // Copy legacy user-created data out before removing any index files.
        await OperationalStore.initialize();
        const status = this.getStatus();
        await OperationalStore.clearRetrievalData();

        clearDirectoryContents(status.logsDir);

        FileSystemService.ensureDirectoryExists(path.join(AppPaths.storageRoot, "runtime"));
        FileSystemService.ensureDirectoryExists(path.join(AppPaths.storageRoot, "logs"));
    }
}

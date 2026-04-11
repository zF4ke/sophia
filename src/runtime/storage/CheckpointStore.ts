import fs from "fs";
import path from "path";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { getAppConfig } from "@/app/AppConfig";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import {
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_VERSION_FILE,
} from "@/runtime/storage/schema";

function deleteSqliteFamily(filePath: string): void {
    for (const suffix of ["", "-wal", "-shm"]) {
        const target = `${filePath}${suffix}`;
        if (fs.existsSync(target)) {
            fs.rmSync(target, { force: true });
        }
    }
}

export class CheckpointStore {
    private static saver: SqliteSaver | null = null;

    public static getSaver(): SqliteSaver {
        if (!this.saver) {
            const checkpointPath = getAppConfig().runtime.checkpointDbPath;
            FileSystemService.ensureDirectoryExists(path.dirname(checkpointPath));
            this.ensureCompatibleStorage(checkpointPath);
            this.saver = SqliteSaver.fromConnString(checkpointPath);
        }

        return this.saver;
    }

    public static async reset(): Promise<void> {
        const saver = this.saver as any;
        await saver?.db?.close?.();
        await saver?.close?.();
        this.saver = null;
    }

    private static ensureCompatibleStorage(checkpointPath: string): void {
        const versionPath = path.join(path.dirname(checkpointPath), CHECKPOINT_VERSION_FILE);
        const currentVersion = fs.existsSync(versionPath)
            ? fs.readFileSync(versionPath, "utf8").trim()
            : null;

        if (currentVersion === CHECKPOINT_SCHEMA_VERSION) {
            return;
        }

        deleteSqliteFamily(checkpointPath);
        fs.writeFileSync(versionPath, CHECKPOINT_SCHEMA_VERSION, "utf8");
    }
}

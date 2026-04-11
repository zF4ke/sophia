import { FileSystemService } from "@/shared/storage/FileSystemService";

const SERVICE_NAME = "debug";
const STATE_FILE = "state.json";

interface DebugStateRecord {
    enabled: boolean;
}

export class DebugStateStore {
    public static load(): DebugStateRecord {
        return (
            FileSystemService.readJsonFromPath<DebugStateRecord>(
                STATE_FILE,
                SERVICE_NAME
            ) ?? { enabled: false }
        );
    }

    public static save(record: DebugStateRecord): void {
        FileSystemService.writeJsonToPath(STATE_FILE, record, SERVICE_NAME);
    }
}


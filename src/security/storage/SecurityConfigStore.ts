import { FileSystemService } from "@/shared/storage/FileSystemService";
import type { AdminUser, CommandConfig, ModeratorUser } from "@/security/types";

const SERVICE_NAME = "security";
const ADMINS_FILE = "admins.json";
const MODERATORS_FILE = "moderators.json";
const COMMANDS_CONFIG_FILE = "commands_config.json";

export class SecurityConfigStore {
    public static loadAdmins(): AdminUser[] | null {
        return FileSystemService.readJsonFromPath<AdminUser[]>(ADMINS_FILE, SERVICE_NAME);
    }

    public static saveAdmins(admins: AdminUser[]): void {
        FileSystemService.writeJsonToPath(ADMINS_FILE, admins, SERVICE_NAME);
    }

    public static loadModerators(): ModeratorUser[] | null {
        return FileSystemService.readJsonFromPath<ModeratorUser[]>(
            MODERATORS_FILE,
            SERVICE_NAME
        );
    }

    public static saveModerators(moderators: ModeratorUser[]): void {
        FileSystemService.writeJsonToPath(MODERATORS_FILE, moderators, SERVICE_NAME);
    }

    public static loadCommandConfigs(): Map<string, CommandConfig> {
        const raw = FileSystemService.readJsonFromPath<Record<string, CommandConfig>>(
            COMMANDS_CONFIG_FILE,
            SERVICE_NAME
        );
        return raw ? new Map(Object.entries(raw)) : new Map();
    }

    public static saveCommandConfigs(commandConfigs: Map<string, CommandConfig>): void {
        FileSystemService.writeJsonToPath(
            COMMANDS_CONFIG_FILE,
            Object.fromEntries(commandConfigs),
            SERVICE_NAME
        );
    }
}


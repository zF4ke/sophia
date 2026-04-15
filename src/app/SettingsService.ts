import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { FileSystemService } from "@/shared/storage/FileSystemService";

export interface BotSettings {
    modelProfile: string;
    runtime: {
        operationalDbPath: string;
        checkpointDbPath: string;
        maxToolCalls: number;
        maxRepeatedCallSignature: number;
        maxLatencyBudgetMs: number;
        maxPriorTurns: number;
        maxChannelMessages: number;
        maxToolRunsContext: number;
        maxEvidenceSlice: number;
        interactiveCrawlLimit: number;
        escalationFetchLimit: number;
        retrievalHistoryLimit: number;
        retrievalContextWindow: number;
        approvalTimeoutMs: number;
        autoApproveWrites: boolean;
    };
    debug: boolean;
}

const SETTINGS_PATH = path.join(AppPaths.storageRoot, "settings.json");
const DEFAULT_RUNTIME_DIR = path.join(AppPaths.storageRoot, "runtime");

const DEFAULT_SETTINGS: BotSettings = {
    modelProfile: "fast",
    runtime: {
        operationalDbPath: path.join(DEFAULT_RUNTIME_DIR, "operational.sqlite"),
        checkpointDbPath: path.join(DEFAULT_RUNTIME_DIR, "checkpoints.sqlite"),
        maxToolCalls: 12,
        maxRepeatedCallSignature: 1,
        maxLatencyBudgetMs: 15000,
        maxPriorTurns: 5,
        maxChannelMessages: 15,
        maxToolRunsContext: 12,
        maxEvidenceSlice: 32,
        interactiveCrawlLimit: 250,
        escalationFetchLimit: 150,
        retrievalHistoryLimit: 50,
        retrievalContextWindow: 15,
        approvalTimeoutMs: 60_000,
        autoApproveWrites: false,
    },
    debug: false,
};

let cached: BotSettings | null = null;

function deepMerge(defaults: BotSettings, overrides: Partial<BotSettings>): BotSettings {
    const result = { ...defaults };
    if (overrides.modelProfile !== undefined) result.modelProfile = overrides.modelProfile;
    if (overrides.debug !== undefined) result.debug = overrides.debug;
    if (overrides.runtime) {
        result.runtime = { ...defaults.runtime, ...overrides.runtime };
    }
    return result;
}

export class SettingsService {
    public static load(): BotSettings {
        if (cached) return cached;

        if (!fs.existsSync(SETTINGS_PATH)) {
            this.save(DEFAULT_SETTINGS);
            cached = { ...DEFAULT_SETTINGS, runtime: { ...DEFAULT_SETTINGS.runtime } };
            return cached;
        }

        try {
            const raw = fs.readFileSync(SETTINGS_PATH, "utf8");
            const parsed = JSON.parse(raw) as Partial<BotSettings>;
            cached = deepMerge(DEFAULT_SETTINGS, parsed);
            return cached;
        } catch {
            cached = { ...DEFAULT_SETTINGS, runtime: { ...DEFAULT_SETTINGS.runtime } };
            return cached;
        }
    }

    public static save(settings: BotSettings): void {
        FileSystemService.ensureDirectoryExists(path.dirname(SETTINGS_PATH));
        fs.writeFileSync(SETTINGS_PATH, JSON.stringify(settings, null, 2), "utf8");
        cached = settings;
    }

    public static update(patch: Partial<BotSettings>): BotSettings {
        const current = this.load();
        const updated = deepMerge(current, patch);
        this.save(updated);
        return updated;
    }

    public static reset(): BotSettings {
        this.save(DEFAULT_SETTINGS);
        return DEFAULT_SETTINGS;
    }

    public static getDefaults(): BotSettings {
        return { ...DEFAULT_SETTINGS, runtime: { ...DEFAULT_SETTINGS.runtime } };
    }

    public static invalidateCache(): void {
        cached = null;
    }
}

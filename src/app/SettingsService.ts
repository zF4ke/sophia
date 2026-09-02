import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { readModelProfiles, resolveModelProfileName } from "@/app/modelProfiles";
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
        escalationFetchLimit: number;
        retrievalHistoryLimit: number;
        retrievalContextWindow: number;
        approvalTimeoutMs: number;
        autoApproveWrites: boolean;
        maxNotesPerRequest: number;
        startupSweep: boolean;
        startupSweepMaxMessages: number;
        edgePrefetch: boolean;
        longTask: {
            maxToolCalls: number;
            maxLatencyBudgetMs: number;
            evidenceSliceFloor: number;
            retrievalInlineCrawlBatches: number;
        };
    };
    compaction: {
        summarizerModel: string;
        triggerFraction: number;
        inputTriggerFraction: number;
    };
    personality: "default" | "mixed" | "classic";
    protectedChannelIds: string[];
    debug: boolean;
}

const SETTINGS_PATH = path.join(AppPaths.storageRoot, "settings.json");
const DEFAULT_RUNTIME_DIR = path.join(AppPaths.storageRoot, "runtime");
const DEFAULT_MODEL_PROFILE = readModelProfiles().defaultProfile;
// Channels that block destructive actions by default. Empty out of the box;
// populate per-deployment via the `/superchannels` command (persisted to the
// git-ignored storage/settings.json).
const DEFAULT_PROTECTED_CHANNEL_IDS: string[] = [];

const DEFAULT_SETTINGS: BotSettings = {
    modelProfile: DEFAULT_MODEL_PROFILE,
    runtime: {
        operationalDbPath: path.join(DEFAULT_RUNTIME_DIR, "operational.sqlite"),
        checkpointDbPath: path.join(DEFAULT_RUNTIME_DIR, "checkpoints.sqlite"),
        maxToolCalls: 25,
        maxRepeatedCallSignature: 1,
        maxLatencyBudgetMs: 120000,
        maxPriorTurns: 8,
        maxChannelMessages: 15,
        maxToolRunsContext: 12,
        maxEvidenceSlice: 32,
        escalationFetchLimit: 600,
        retrievalHistoryLimit: 1000,
        retrievalContextWindow: 15,
        approvalTimeoutMs: 60_000,
        autoApproveWrites: false,
        maxNotesPerRequest: 200,
        startupSweep: true,
        startupSweepMaxMessages: 1000,
        edgePrefetch: true,
        longTask: {
            maxToolCalls: 200,
            maxLatencyBudgetMs: 600_000,
            evidenceSliceFloor: 128,
            retrievalInlineCrawlBatches: 3,
        },
    },
    compaction: {
        summarizerModel: "glm53flash",
        triggerFraction: 0.88,
        inputTriggerFraction: 0.55,
    },
    personality: "default",
    protectedChannelIds: DEFAULT_PROTECTED_CHANNEL_IDS,
    debug: false,
};

let cached: BotSettings | null = null;

function deepMerge(defaults: BotSettings, overrides: Partial<BotSettings>): BotSettings {
    const result = { ...defaults };
    if (overrides.modelProfile !== undefined) result.modelProfile = overrides.modelProfile;
    if (overrides.personality !== undefined) {
        const allowed: BotSettings["personality"][] = ["default", "mixed", "classic"];
        result.personality = allowed.includes(overrides.personality)
            ? overrides.personality
            : defaults.personality;
    }
    if (overrides.protectedChannelIds !== undefined) result.protectedChannelIds = overrides.protectedChannelIds;
    if (overrides.debug !== undefined) result.debug = overrides.debug;
    if (overrides.runtime) {
        const mergedRuntime = { ...defaults.runtime, ...overrides.runtime };
        mergedRuntime.longTask = {
            ...defaults.runtime.longTask,
            ...((overrides.runtime as Partial<BotSettings["runtime"]>).longTask ?? {}),
        };
        result.runtime = mergedRuntime;
    }
    if (overrides.compaction) {
        result.compaction = { ...defaults.compaction, ...overrides.compaction };
    }
    return result;
}

export class SettingsService {
    public static load(): BotSettings {
        if (cached) return cached;

        if (!fs.existsSync(SETTINGS_PATH)) {
            this.save(DEFAULT_SETTINGS);
            cached = { ...DEFAULT_SETTINGS, runtime: { ...DEFAULT_SETTINGS.runtime, longTask: { ...DEFAULT_SETTINGS.runtime.longTask } }, compaction: { ...DEFAULT_SETTINGS.compaction } };
            return cached;
        }

        try {
            const raw = fs.readFileSync(SETTINGS_PATH, "utf8");
            const parsed = JSON.parse(raw) as Partial<BotSettings>;
            const merged = deepMerge(DEFAULT_SETTINGS, parsed);
            const profileConfig = readModelProfiles();
            merged.modelProfile = resolveModelProfileName(merged.modelProfile, profileConfig);

            // Normalize compaction model: fall back to default if the saved value is invalid.
            if (!profileConfig.profiles[merged.compaction.summarizerModel]) {
                merged.compaction.summarizerModel = DEFAULT_SETTINGS.compaction.summarizerModel;
            }
            cached = merged;

            const needsSave =
                parsed.modelProfile !== merged.modelProfile ||
                (parsed.compaction as Partial<BotSettings["compaction"]> | undefined)?.summarizerModel !== merged.compaction.summarizerModel ||
                // Backfill newly-added runtime knobs into the on-disk file so
                // operators can see (and tune) them without reading source.
                (parsed.runtime as Partial<BotSettings["runtime"]> | undefined)?.startupSweep === undefined ||
                (parsed.runtime as Partial<BotSettings["runtime"]> | undefined)?.startupSweepMaxMessages === undefined ||
                (parsed.runtime as Partial<BotSettings["runtime"]> | undefined)?.edgePrefetch === undefined;
            if (needsSave) {
                this.save(merged);
            }

            return cached;
        } catch {
            cached = { ...DEFAULT_SETTINGS, runtime: { ...DEFAULT_SETTINGS.runtime, longTask: { ...DEFAULT_SETTINGS.runtime.longTask } }, compaction: { ...DEFAULT_SETTINGS.compaction } };
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
        return { ...DEFAULT_SETTINGS, runtime: { ...DEFAULT_SETTINGS.runtime, longTask: { ...DEFAULT_SETTINGS.runtime.longTask } }, compaction: { ...DEFAULT_SETTINGS.compaction } };
    }

    public static invalidateCache(): void {
        cached = null;
    }
}

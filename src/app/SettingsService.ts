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
const DEFAULT_PROTECTED_CHANNEL_IDS = [
    "731284154066534521",
    "731276952882642947",
    "731276952882642945",
    "826976703641026611",
    "731278507740495882",
    "739241649791697006",
    "1413575314251972740",
    "741780701443129558",
    "731576766451154944",
    "1180114828128419920",
    "1240415671813279817",
    "1359602913877557248",
    "1179466942583685150",
    "1179467152156274758",
    "1179467188701245470",
    "1179467222058532944",
    "1179467395467841649",
    "1233966827130847382",
    "1233966854611931180",
    "888219991239962654",
    "731517713180131380",
    "764880116849311815",
    "756559296862617720",
    "764696949852864512",
    "764875079019921418",
    "731277002174103573",
    "731284421252218985",
    "731286303089623133",
    "731286328968347718",
    "745128654073495630",
    "745431693829341184",
    "767127675332067378",
    "767472790194356285",
    "767487063637229569",
    "767492797254729778",
    "749690989756809266",
    "733913987443589131",
    "743817000983462079",
    "742099856293756999",
    "749690861469827216",
    "738977919342477363",
    "738945304887558234",
    "787411009181712415",
];

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
        longTask: {
            maxToolCalls: 200,
            maxLatencyBudgetMs: 600_000,
            evidenceSliceFloor: 128,
            retrievalInlineCrawlBatches: 3,
        },
    },
    compaction: {
        summarizerModel: "gemini31flashlite",
        triggerFraction: 0.85,
        inputTriggerFraction: 0.4,
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
                (parsed.compaction as Partial<BotSettings["compaction"]> | undefined)?.summarizerModel !== merged.compaction.summarizerModel;
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

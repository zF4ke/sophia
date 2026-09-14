import fs from "fs";
import path from "path";
import { randomUUID } from "node:crypto";
import { AppPaths } from "@/app/AppPaths";
import { readModelProfiles, resolveModelProfileName } from "@/app/modelProfiles";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import { accessConfigSchema, type AccessConfig } from "@/security/accessConfig";
import { z } from "zod";
import { MigrationBackup } from "@/shared/storage/MigrationBackup";

export interface BotSettings {
    schemaVersion: 5;
    sandbox: { enabled: boolean; image: string };
    scheduling: { enabled: boolean; pollIntervalMs: number };
    memory: { dreamingEnabled: boolean; dreamIntervalMs: number };
    access: AccessConfig;
    modelProfile: string;
    runtime: {
        operationalDbPath: string;
        toolCallLimit: number;
        modelConcurrency: number;
        maxRepeatedCallSignature: number;
        maxPriorTurns: number;
        maxChannelMessages: number;
        maxToolRunsContext: number;
        maxEvidenceSlice: number;
        escalationFetchLimit: number;
        retrievalHistoryLimit: number;
        retrievalContextWindow: number;
        approvalTimeoutMs: number;
        startupSweep: boolean;
        startupSweepMaxMessages: number;
        edgePrefetch: boolean;
        longTask: {
            evidenceSliceFloor: number;
        };
    };
    compaction: {
        summarizerModel: string;
        triggerFraction: number;
        inputTriggerFraction: number;
    };
    voice: "balanced" | "casual" | "formal";
    protectedChannelIds: string[];
    /** Explicit enabled guilds. Empty enables no guilds. DMs are configured separately. */
    guildAllowlist: string[];
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
    schemaVersion: 5,
    sandbox: { enabled: true, image: "sophia-sandbox:5" },
    scheduling: { enabled: true, pollIntervalMs: 15_000 },
    memory: { dreamingEnabled: true, dreamIntervalMs: 300_000 },
    access: accessConfigSchema.parse({}),
    modelProfile: DEFAULT_MODEL_PROFILE,
    runtime: {
        operationalDbPath: path.join(DEFAULT_RUNTIME_DIR, "operational.sqlite"),
        toolCallLimit: 0,
        modelConcurrency: 4,
        maxRepeatedCallSignature: 1,
        maxPriorTurns: 8,
        maxChannelMessages: 15,
        maxToolRunsContext: 12,
        maxEvidenceSlice: 32,
        escalationFetchLimit: 600,
        retrievalHistoryLimit: 1000,
        retrievalContextWindow: 15,
        approvalTimeoutMs: 60_000,
        startupSweep: true,
        startupSweepMaxMessages: 1000,
        edgePrefetch: true,
        longTask: {
            evidenceSliceFloor: 128,
        },
    },
    compaction: {
        summarizerModel: DEFAULT_MODEL_PROFILE,
        triggerFraction: 0.88,
        inputTriggerFraction: 0.55,
    },
    voice: "balanced",
    protectedChannelIds: DEFAULT_PROTECTED_CHANNEL_IDS,
    guildAllowlist: [],
    debug: false,
};

let cached: BotSettings | null = null;

function deepMerge(defaults: BotSettings, overrides: Partial<BotSettings>): BotSettings {
    const result = { ...defaults };
    if (overrides.schemaVersion !== undefined && overrides.schemaVersion !== 5) throw new Error("Unsupported settings schema version.");
    result.schemaVersion = 5;
    result.sandbox = { ...defaults.sandbox, ...overrides.sandbox };
    result.scheduling = { ...defaults.scheduling, ...overrides.scheduling };
    if (typeof result.scheduling.enabled !== "boolean" || !Number.isSafeInteger(result.scheduling.pollIntervalMs) || result.scheduling.pollIntervalMs < 1000) throw new Error("Invalid scheduling settings.");
    if (typeof result.sandbox.enabled !== "boolean" || typeof result.sandbox.image !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9./:@_-]+$/.test(result.sandbox.image)) throw new Error("Invalid sandbox settings.");
    result.memory = { ...defaults.memory, ...overrides.memory };
    if (typeof result.memory.dreamingEnabled !== "boolean" || !Number.isSafeInteger(result.memory.dreamIntervalMs) || result.memory.dreamIntervalMs < 60_000) {
        throw new Error("Invalid memory settings; dreamIntervalMs must be at least 60000.");
    }
    if (overrides.modelProfile !== undefined) result.modelProfile = overrides.modelProfile;
    const legacyPersonality = (overrides as unknown as { personality?: string }).personality;
    result.voice = overrides.voice ?? (legacyPersonality === "mixed" ? "casual" : legacyPersonality === "classic" ? "formal" : defaults.voice);
    if (!["balanced", "casual", "formal"].includes(result.voice)) throw new Error("Invalid voice setting.");
    if (overrides.protectedChannelIds !== undefined) result.protectedChannelIds = overrides.protectedChannelIds;
    if (overrides.guildAllowlist !== undefined) result.guildAllowlist = overrides.guildAllowlist;
    if (!Array.isArray(result.guildAllowlist) || result.guildAllowlist.some(id => typeof id !== "string" || !id.trim())) {
        throw new Error("guildAllowlist must contain non-empty guild IDs");
    }
    result.access = accessConfigSchema.parse(overrides.access ?? defaults.access);
    if (overrides.debug !== undefined) result.debug = overrides.debug;
    if (overrides.runtime) {
        const mergedRuntime = { ...defaults.runtime, ...overrides.runtime };
        mergedRuntime.longTask = {
            ...defaults.runtime.longTask,
            ...((overrides.runtime as Partial<BotSettings["runtime"]>).longTask ?? {}),
        };
        // Retire v4 call budgets instead of silently making them v5 limits.
        delete (mergedRuntime as unknown as Record<string, unknown>).maxToolCalls;
        delete (mergedRuntime as unknown as Record<string, unknown>).checkpointDbPath;
        delete (mergedRuntime as unknown as Record<string, unknown>).autoApproveWrites;
        delete (mergedRuntime as unknown as Record<string, unknown>).maxNotesPerRequest;
        delete (mergedRuntime.longTask as unknown as Record<string, unknown>).maxToolCalls;
        delete (mergedRuntime.longTask as unknown as Record<string, unknown>).retrievalInlineCrawlBatches;
        if (!Number.isSafeInteger(mergedRuntime.toolCallLimit) || mergedRuntime.toolCallLimit < 0) {
            throw new Error("runtime.toolCallLimit must be a non-negative integer");
        }
        result.runtime = mergedRuntime;
    }
    if (overrides.compaction) {
        result.compaction = { ...defaults.compaction, ...overrides.compaction };
    }
    z.object({ triggerFraction: z.number().gt(0).lt(1), inputTriggerFraction: z.number().gt(0).lt(1), summarizerModel: z.string().min(1) }).parse(result.compaction);
    z.boolean().parse(result.debug);
    z.array(z.string().min(1)).parse(result.protectedChannelIds);
    const rt = result.runtime;
    z.number().int().min(1).max(32).parse(rt.modelConcurrency);
    z.string().min(1).parse(rt.operationalDbPath);
    for (const [key, value] of Object.entries(rt)) {
        if (typeof value === "number") {
            if (!Number.isSafeInteger(value) || value < 0) throw new Error(`runtime.${key} must be a non-negative integer.`);
        } else if (!["operationalDbPath", "longTask", "startupSweep", "edgePrefetch"].includes(key)) {
            throw new Error(`Invalid runtime setting: ${key}.`);
        }
    }
    for (const key of ["startupSweep", "edgePrefetch"] as const) z.boolean().parse(rt[key]);
    z.object({ evidenceSliceFloor: z.number().int().nonnegative() }).parse(rt.longTask);
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

            const legacyRuntime = parsed.runtime as unknown as Record<string, unknown> | undefined;
            const needsSave =
                legacyRuntime?.maxNotesPerRequest !== undefined ||
                legacyRuntime?.autoApproveWrites !== undefined ||
                parsed.schemaVersion !== 5 ||
                legacyRuntime?.checkpointDbPath !== undefined ||
                parsed.voice === undefined ||
                "personality" in parsed ||
                parsed.memory === undefined ||
                parsed.sandbox === undefined ||
                parsed.scheduling === undefined ||
                parsed.access === undefined ||
                legacyRuntime?.maxToolCalls !== undefined ||
                (legacyRuntime?.longTask as Record<string, unknown> | undefined)?.retrievalInlineCrawlBatches !== undefined ||
                (legacyRuntime?.longTask as Record<string, unknown> | undefined)?.maxToolCalls !== undefined ||
                legacyRuntime?.toolCallLimit === undefined ||
                parsed.modelProfile !== merged.modelProfile ||
                (parsed.compaction as Partial<BotSettings["compaction"]> | undefined)?.summarizerModel !== merged.compaction.summarizerModel ||
                // Backfill newly-added runtime knobs into the on-disk file so
                // operators can see (and tune) them without reading source.
                (parsed.runtime as Partial<BotSettings["runtime"]> | undefined)?.startupSweep === undefined ||
                (parsed.runtime as Partial<BotSettings["runtime"]> | undefined)?.startupSweepMaxMessages === undefined ||
                (parsed.runtime as Partial<BotSettings["runtime"]> | undefined)?.edgePrefetch === undefined;
            if (needsSave) {
                MigrationBackup.file(SETTINGS_PATH);
                this.save(merged);
            }

            return cached;
        } catch (error) {
            cached = null;
            throw Object.assign(new Error("Unable to load settings. Repair settings.json before starting Sophia."), { cause: error });
        }
    }

    public static save(settings: BotSettings): void {
        FileSystemService.ensureDirectoryExists(path.dirname(SETTINGS_PATH));
        const temporaryPath = `${SETTINGS_PATH}.${randomUUID()}.tmp`;
        try {
            fs.writeFileSync(temporaryPath, JSON.stringify(settings, null, 2), "utf8");
            // Windows file scanners can briefly hold the destination open.
            // Retry the atomic replacement; never delete the working settings.
            for (let attempt = 0; ; attempt++) {
                try {
                    fs.renameSync(temporaryPath, SETTINGS_PATH);
                    break;
                } catch (error) {
                    const code = (error as NodeJS.ErrnoException).code;
                    if (process.platform !== "win32" || !["EPERM", "EBUSY", "EACCES"].includes(code ?? "") || attempt >= 5) throw error;
                    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10 * 2 ** attempt);
                }
            }
        } finally {
            if (fs.existsSync(temporaryPath)) fs.unlinkSync(temporaryPath);
        }
        cached = settings;
    }

    public static update(patch: Partial<BotSettings>): BotSettings {
        const current = this.load();
        const updated = deepMerge(current, patch);
        const profiles = readModelProfiles();
        updated.modelProfile = resolveModelProfileName(updated.modelProfile, profiles);
        if (!updated.compaction || !profiles.profiles[updated.compaction.summarizerModel]) {
            updated.compaction = {
                ...DEFAULT_SETTINGS.compaction,
                ...(updated.compaction ?? {}),
                summarizerModel: profiles.defaultProfile,
            };
        }
        this.save(updated);
        return updated;
    }

    public static reset(category: "runtime" | "model" | "voice" | "compaction" | "memory" | "sandbox" | "scheduling" = "runtime"): BotSettings {
        const defaults = this.getDefaults();
        if (category === "model") return this.update({ modelProfile: defaults.modelProfile });
        if (category === "runtime") return this.update({ runtime: { ...defaults.runtime, operationalDbPath: this.load().runtime.operationalDbPath } });
        return this.update({ [category]: defaults[category] });
    }

    public static getDefaults(): BotSettings {
        return structuredClone(DEFAULT_SETTINGS);
    }

    public static invalidateCache(): void {
        cached = null;
    }
}

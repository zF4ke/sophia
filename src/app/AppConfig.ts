import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { SettingsService } from "@/app/SettingsService";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import type { AppConfig, ModelProfileConfig } from "@/shared/appTypes";

function readModelProfiles(): ModelProfileConfig {
    const raw = fs.readFileSync(AppPaths.modelProfilesPath, "utf8");
    const parsed = JSON.parse(raw) as ModelProfileConfig;

    if (!parsed.defaultProfile || !parsed.profiles?.[parsed.defaultProfile]) {
        throw new Error("Invalid model profile configuration.");
    }

    return parsed;
}

export function getAppConfig(): AppConfig {
    const profiles = readModelProfiles();
    const settings = SettingsService.load();

    const modelProfileName = settings.modelProfile || profiles.defaultProfile;
    const modelProfile = profiles.profiles[modelProfileName];

    if (!modelProfile) {
        throw new Error(`Unknown model profile "${modelProfileName}".`);
    }

    const discordToken = process.env.DISCORD_TOKEN;
    const openRouterApiKey = process.env.OPENROUTER_API_KEY;

    if (!discordToken) {
        throw new Error("Missing DISCORD_TOKEN.");
    }

    if (!openRouterApiKey) {
        throw new Error("Missing OPENROUTER_API_KEY.");
    }

    FileSystemService.ensureDirectoryExists(FileSystemService.getBaseStorageDir());
    const runtimeDir = path.join(AppPaths.storageRoot, "runtime");
    FileSystemService.ensureDirectoryExists(runtimeDir);

    const rt = settings.runtime;

    return {
        discordToken,
        openRouterApiKey,
        openRouterBaseUrl: "https://openrouter.ai/api/v1",
        modelProfileName,
        modelProfile,
        runtime: {
            operationalDbPath: rt.operationalDbPath,
            checkpointDbPath: rt.checkpointDbPath,
            maxToolCalls: rt.maxToolCalls,
            maxRepeatedCallSignature: rt.maxRepeatedCallSignature,
            maxLatencyBudgetMs: rt.maxLatencyBudgetMs,
            maxPriorTurns: rt.maxPriorTurns,
            maxChannelMessages: rt.maxChannelMessages,
            maxToolRunsContext: rt.maxToolRunsContext,
            maxEvidenceSlice: rt.maxEvidenceSlice,
            escalationFetchLimit: rt.escalationFetchLimit,
            retrievalHistoryLimit: rt.retrievalHistoryLimit,
            retrievalContextWindow: rt.retrievalContextWindow,
        },
    };
}

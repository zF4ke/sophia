import path from "path";
import { SettingsService } from "@/app/SettingsService";
import { readModelProfiles, resolveModelProfileName } from "@/app/modelProfiles";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import type { AppConfig } from "@/shared/appTypes";

export function getAppConfig(): AppConfig {
    const profiles = readModelProfiles();
    const settings = SettingsService.load();

    const modelProfileName = resolveModelProfileName(settings.modelProfile, profiles);
    const modelProfile = profiles.profiles[modelProfileName];

    if (!modelProfile) {
        throw new Error(`Unknown model profile "${modelProfileName}".`);
    }

    if (settings.modelProfile !== modelProfileName) {
        SettingsService.update({ modelProfile: modelProfileName });
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
    const runtimeDir = path.join(FileSystemService.getBaseStorageDir(), "runtime");
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
            longTask: {
                maxToolCalls: rt.longTask?.maxToolCalls ?? 200,
                maxLatencyBudgetMs: rt.longTask?.maxLatencyBudgetMs ?? 600_000,
                evidenceSliceFloor: rt.longTask?.evidenceSliceFloor ?? 128,
                retrievalInlineCrawlBatches: rt.longTask?.retrievalInlineCrawlBatches ?? 3,
            },
        },
    };
}

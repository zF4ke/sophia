import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
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
    const modelProfileName = process.env.MODEL_PROFILE || profiles.defaultProfile;
    const modelProfile = profiles.profiles[modelProfileName];

    if (!modelProfile) {
        throw new Error(`Unknown model profile "${modelProfileName}".`);
    }

    const discordToken = process.env.DISCORD_TOKEN || process.env.CLIENT_TOKEN;
    const openRouterApiKey = process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY;

    if (!discordToken) {
        throw new Error("Missing DISCORD_TOKEN.");
    }

    if (!openRouterApiKey) {
        throw new Error("Missing OPENROUTER_API_KEY.");
    }

    FileSystemService.ensureDirectoryExists(FileSystemService.getBaseStorageDir());
    const runtimeDir = path.join(AppPaths.storageRoot, "runtime");
    FileSystemService.ensureDirectoryExists(runtimeDir);

    return {
        discordToken,
        openRouterApiKey,
        openRouterBaseUrl:
            process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
        modelProfileName,
        modelProfile,
        runtime: {
            operationalDbPath:
                process.env.RUNTIME_OPERATIONAL_DB_PATH ||
                path.join(runtimeDir, "operational.sqlite"),
            checkpointDbPath:
                process.env.RUNTIME_CHECKPOINT_DB_PATH ||
                path.join(runtimeDir, "checkpoints.sqlite"),
            maxToolCalls: Number(process.env.RUNTIME_MAX_TOOL_CALLS || 6),
            maxResearchPasses: Number(process.env.RUNTIME_MAX_RESEARCH_PASSES || 4),
            maxRepeatedCallSignature: Number(
                process.env.RUNTIME_MAX_REPEATED_CALL_SIGNATURE || 1
            ),
            maxLatencyBudgetMs: Number(process.env.RUNTIME_MAX_LATENCY_BUDGET_MS || 15000),
        },
    };
}

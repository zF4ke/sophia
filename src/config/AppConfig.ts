import fs from "fs";
import path from "path";
import { FileSystemService } from "@/services/FileSystemService";
import type { AppConfig, ModelProfileConfig } from "@/types/app";

const MODEL_PROFILE_PATH = path.join(process.cwd(), "config", "model-profiles.json");

function readModelProfiles(): ModelProfileConfig {
    const raw = fs.readFileSync(MODEL_PROFILE_PATH, "utf8");
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

    FileSystemService.ensureDirectoryExists(FileSystemService.getBaseDataDir());

    return {
        discordToken,
        openRouterApiKey,
        openRouterBaseUrl:
            process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
        port: Number(process.env.PORT || 3002),
        modelProfileName,
        modelProfile,
    };
}

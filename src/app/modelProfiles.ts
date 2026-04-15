import fs from "fs";
import { AppPaths } from "@/app/AppPaths";
import type { ModelProfileConfig } from "@/shared/appTypes";

const FALLBACK_MODEL_PROFILES: ModelProfileConfig = {
    defaultProfile: "free-elephant",
    profiles: {
        gemini31flashlite: {
            chatModel: "google/gemini-3.1-flash-lite-preview",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 1_000_000,
            parallelToolCalls: false,
        },
        minimax27: {
            chatModel: "minimax/minimax-m2.7",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1400,
            contextWindow: 190_000,
            parallelToolCalls: false,
        },
        deepseek32: {
            chatModel: "deepseek/deepseek-v3.2",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1400,
            contextWindow: 160_000,
            parallelToolCalls: false,
        },
        "free-nvidia": {
            chatModel: "nvidia/nemotron-3-nano-30b-a3b:free",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 131_072,
            parallelToolCalls: false,
        },
        "free-elephant": {
            chatModel: "openrouter/elephant-alpha",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 260_000,
            parallelToolCalls: false,
        },
        grok41: {
            chatModel: "x-ai/grok-4.1-fast",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 2_000_000,
            parallelToolCalls: false,
        },
    },
};

const LEGACY_MODEL_PROFILE_ALIASES: Record<string, string> = {
    fast: "gemini31flashlite",
    smarter: "minimax27",
    alt: "deepseek32",
    cheap: "free-elephant",
};

function isValidModelProfileConfig(value: unknown): value is ModelProfileConfig {
    if (!value || typeof value !== "object") return false;

    const candidate = value as Partial<ModelProfileConfig>;
    if (!candidate.defaultProfile || typeof candidate.defaultProfile !== "string") return false;
    if (!candidate.profiles || typeof candidate.profiles !== "object") return false;

    return Boolean(candidate.profiles[candidate.defaultProfile]);
}

export function readModelProfiles(): ModelProfileConfig {
    try {
        const raw = fs.readFileSync(AppPaths.modelProfilesPath, "utf8");
        const parsed = JSON.parse(raw) as unknown;

        if (isValidModelProfileConfig(parsed)) {
            return parsed;
        }
    } catch {
        // Fall back to the checked-in profile set if the file is missing or invalid.
    }

    return FALLBACK_MODEL_PROFILES;
}

export function resolveModelProfileName(
    requestedName: string | null | undefined,
    config: ModelProfileConfig = readModelProfiles(),
): string {
    const normalizedName = requestedName?.trim();

    if (normalizedName && config.profiles[normalizedName]) {
        return normalizedName;
    }

    const legacyAlias = normalizedName ? LEGACY_MODEL_PROFILE_ALIASES[normalizedName] : undefined;
    if (legacyAlias && config.profiles[legacyAlias]) {
        return legacyAlias;
    }

    return config.defaultProfile;
}
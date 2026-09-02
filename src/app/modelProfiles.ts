import fs from "fs";
import { AppPaths } from "@/app/AppPaths";
import type { ModelProfileConfig } from "@/shared/appTypes";

const FALLBACK_MODEL_PROFILES: ModelProfileConfig = {
    defaultProfile: "glm53flash",
    profiles: {
        glm53flash: {
            label: "GLM 5.3 Flash",
            chatModel: "z-ai/glm-5.3-flash",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.45,
            maxOutputTokens: 12000,
            contextWindow: 1_048_576,
            parallelToolCalls: true,
        },
        gemini31flashlite: {
            label: "Gemini 3.1 Flash Lite",
            chatModel: "google/gemini-3.1-flash-lite-preview",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 1_000_000,
            parallelToolCalls: false,
        },
        minimax27: {
            label: "MiniMax M2.7",
            chatModel: "minimax/minimax-m2.7",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1400,
            contextWindow: 190_000,
            parallelToolCalls: false,
        },
        deepseek32: {
            label: "DeepSeek V3.2",
            chatModel: "deepseek/deepseek-v3.2",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1400,
            contextWindow: 160_000,
            parallelToolCalls: false,
        },
        "free-nvidia": {
            label: "Nemotron 3 Nano (free)",
            chatModel: "nvidia/nemotron-3-nano-30b-a3b:free",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 131_072,
            parallelToolCalls: false,
        },
        "free-elephant": {
            label: "Elephant Alpha (free)",
            chatModel: "openrouter/elephant-alpha",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 260_000,
            parallelToolCalls: false,
        },
        grok41: {
            label: "Grok 4.1 Fast",
            chatModel: "x-ai/grok-4.1-fast",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 2_000_000,
            parallelToolCalls: false,
            pricing: {
                inputPerMillionUsd: 0.2,
                outputPerMillionUsd: 0.5,
                source: "openrouter",
                updatedAt: "2026-04-16",
            },
        },
        gpt5nano: {
            label: "GPT-5 Nano",
            chatModel: "openai/gpt-5-nano",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 400_000,
            parallelToolCalls: false,
            pricing: {
                inputPerMillionUsd: 0.05,
                outputPerMillionUsd: 0.4,
                cacheReadPerMillionUsd: 0.01,
                source: "openrouter",
                updatedAt: "2026-04-16",
            },
        },
        gpt41nano: {
            label: "GPT-4.1 Nano",
            chatModel: "openai/gpt-4.1-nano",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 1_000_000,
            parallelToolCalls: false,
            pricing: {
                inputPerMillionUsd: 0.1,
                outputPerMillionUsd: 0.4,
                cacheReadPerMillionUsd: 0.025,
                webSearchPerCallUsd: 0.01,
                source: "openrouter",
                updatedAt: "2026-04-16",
            },
        },
        gpt54nano: {
            label: "GPT-5.4 Nano",
            chatModel: "openai/gpt-5.4-nano",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.5,
            maxOutputTokens: 1200,
            contextWindow: 400_000,
            parallelToolCalls: false,
            pricing: {
                inputPerMillionUsd: 0.2,
                outputPerMillionUsd: 1.25,
                cacheReadPerMillionUsd: 0.02,
                webSearchPerCallUsd: 0.01,
                source: "openrouter",
                updatedAt: "2026-04-16",
            },
        },
    },
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

    return config.defaultProfile;
}

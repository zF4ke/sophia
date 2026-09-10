import fs from "fs";
import { AppPaths } from "@/app/AppPaths";
import type { ModelProfile, ModelProfileConfig } from "@/shared/appTypes";

const EMERGENCY_MODEL_PROFILES: ModelProfileConfig = {
    defaultProfile: "glm53flash",
    profiles: {
        glm53flash: {
            label: "GLM 5.3 Flash",
            chatModel: "z-ai/glm-5.3-flash",
            embeddingModel: "openai/text-embedding-3-small",
            temperature: 0.45,
            maxOutputTokens: 24_000,
            contextWindow: 1_310_720,
            parallelToolCalls: true,
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

    return EMERGENCY_MODEL_PROFILES;
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

export function listModelProfiles(
    config: ModelProfileConfig = readModelProfiles(),
): Array<[string, ModelProfile]> {
    return Object.entries(config.profiles).sort(([leftName, left], [rightName, right]) => {
        const leftInput = left.pricing?.inputPerMillionUsd ?? Number.POSITIVE_INFINITY;
        const rightInput = right.pricing?.inputPerMillionUsd ?? Number.POSITIVE_INFINITY;
        if (leftInput !== rightInput) return leftInput - rightInput;

        const leftOutput = left.pricing?.outputPerMillionUsd ?? Number.POSITIVE_INFINITY;
        const rightOutput = right.pricing?.outputPerMillionUsd ?? Number.POSITIVE_INFINITY;
        if (leftOutput !== rightOutput) return leftOutput - rightOutput;

        return (left.label || leftName).localeCompare(right.label || rightName);
    });
}

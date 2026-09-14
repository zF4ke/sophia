import fs from "fs";
import { AppPaths } from "@/app/AppPaths";
import type { ModelProfile, ModelProfileConfig } from "@/shared/appTypes";
import { z } from "zod";

const profileSchema = z.object({
    api: z.enum(["chat-completions", "responses"]).optional(),
    reasoningEffort: z.enum(["minimal", "low", "medium", "high"]).optional(),
    label: z.string().optional(), chatModel: z.string().trim().min(1), embeddingModel: z.string().trim().min(1),
    temperature: z.number().finite().min(0).max(2), maxOutputTokens: z.number().int().positive(),
    contextWindow: z.number().int().positive(), parallelToolCalls: z.boolean().optional(),
    inputModalities: z.array(z.enum(["text", "image", "video", "audio"])).min(1).optional(),
    baseUrl: z.string().url().refine(value => {
        const url = new URL(value);
        return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password;
    }, "Use an HTTP(S) endpoint without embedded credentials").optional(),
    apiKeyEnv: z.string().regex(/^[A-Za-z_][A-Za-z0-9_]*$/).optional(),
    provider: z.string().optional(), notes: z.string().optional(),
    pricing: z.object({ inputPerMillionUsd: z.number().nonnegative().optional(), outputPerMillionUsd: z.number().nonnegative().optional(),
        cacheReadPerMillionUsd: z.number().nonnegative().optional(), webSearchPerCallUsd: z.number().nonnegative().optional(),
        source: z.string().optional(), updatedAt: z.string().optional() }).optional(),
}).refine(profile => profile.maxOutputTokens < profile.contextWindow, "Output tokens must leave room for input");

export function parseModelProfiles(value: unknown): ModelProfileConfig {
    const config = z.object({ defaultProfile: z.string().min(1), profiles: z.record(z.string(), profileSchema) }).parse(value);
    if (!Object.prototype.hasOwnProperty.call(config.profiles, config.defaultProfile)) throw new Error("Default model profile does not exist.");
    return config;
}

export function readModelProfiles(): ModelProfileConfig {
    try {
        const raw = fs.readFileSync(AppPaths.modelProfilesPath, "utf8");
        const parsed = JSON.parse(raw) as unknown;

        return parseModelProfiles(parsed);
    } catch (error) {
        throw new Error(`Unable to load model profiles from ${AppPaths.modelProfilesPath}. Correct the configuration before starting: ${error instanceof Error ? error.message : String(error)}`);
    }
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

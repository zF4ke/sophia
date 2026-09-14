import { vi } from "vitest";
import { ModelGateway } from "@/ai/ModelGateway";
import { SettingsService } from "@/app/SettingsService";
import { readModelProfiles } from "@/app/modelProfiles";
import { countTokens } from "@/shared/tokenizer";

/** Explicit bounds for opt-in tests, independent of production execution policy. */
export function boundLiveModelCalls(maxCalls = 6, estimatedLimitUsd = 0.10) {
    if (process.env.LIVE_MODEL_PROFILE) {
        if (!readModelProfiles().profiles[process.env.LIVE_MODEL_PROFILE]) throw new Error("Unknown live test profile.");
        SettingsService.update({ modelProfile: process.env.LIVE_MODEL_PROFILE });
    }
    const current = SettingsService.load();
    const profile = readModelProfiles().profiles[current.modelProfile];
    const inputPrice = profile?.pricing?.inputPerMillionUsd;
    const outputPrice = profile?.pricing?.outputPerMillionUsd;
    if (!profile || typeof inputPrice !== "number" || typeof outputPrice !== "number" || !Number.isFinite(inputPrice + outputPrice)) throw new Error("Live tests require configured model prices.");
    SettingsService.update({ runtime: { ...current.runtime, toolCallLimit: maxCalls } });
    let calls = 0, estimatedCeilingUsd = 0;
    const generate = ModelGateway.generateWithTools.bind(ModelGateway);
    vi.spyOn(ModelGateway, "generateWithTools").mockImplementation(async (messages, options) => {
        if (++calls > maxCalls) throw new Error("Live test request limit reached.");
        const inputTokens = countTokens(JSON.stringify({ messages, tools: options.tools })) + 1000;
        estimatedCeilingUsd += 3 * (inputTokens * inputPrice + 2048 * outputPrice) / 1_000_000;
        if (estimatedCeilingUsd > estimatedLimitUsd) throw new Error("Live test estimated spending ceiling reached.");
        return generate(messages, { ...options, maxOutputTokens: 2048 });
    });
    return () => ({ model: profile.chatModel, calls, estimatedCeilingUsd });
}

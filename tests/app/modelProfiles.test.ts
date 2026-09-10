import { describe, expect, it } from "vitest";
import {
    listModelProfiles,
    readModelProfiles,
    resolveModelProfileName,
} from "@/app/modelProfiles";

describe("model profiles", () => {
    it("exposes only the supported profiles", () => {
        const config = readModelProfiles();

        expect(Object.keys(config.profiles).sort()).toEqual([
            "glm53flash",
            "gptoss120b",
            "ling30flash",
            "localLmStudio",
            "museSpark12Zen",
            "museSpark13Contributor",
        ]);
    });

    it("uses one ordering for every model picker", () => {
        expect(listModelProfiles().map(([name]) => name)).toEqual([
            "localLmStudio",
            "museSpark12Zen",
            "ling30flash",
            "gptoss120b",
            "glm53flash",
            "museSpark13Contributor",
        ]);
    });

    it("falls back to the canonical default for removed profiles", () => {
        expect(resolveModelProfileName("minimax27")).toBe("glm53flash");
    });

    it("points the local profile at an OpenAI-compatible local server", () => {
        const config = readModelProfiles();
        const local = config.profiles.localLmStudio;

        expect(local.baseUrl).toBe("http://127.0.0.1:1234/v1");
        expect(local.pricing?.inputPerMillionUsd).toBe(0);
    });
});

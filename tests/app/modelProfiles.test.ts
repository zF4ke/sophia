import { describe, expect, it } from "vitest";
import { parseModelProfiles, readModelProfiles } from "@/app/modelProfiles";

describe("model profile configuration", () => {
    const profile = { chatModel: "configured/model", embeddingModel: "configured/embedding", temperature: 0.5, maxOutputTokens: 2000, contextWindow: 8000 };
    const parse = (overrides = {}) => parseModelProfiles({ defaultProfile: "configured", profiles: { configured: { ...profile, ...overrides } } });
    it("loads checked-in profiles without a source-code fallback", () => {
        expect(readModelProfiles().defaultProfile).toBeTruthy();
        expect(parse({ baseUrl: "http://127.0.0.1:1234/v1", inputModalities: ["text", "image"] }).profiles.configured.baseUrl).toContain("1234");
    });
    it("rejects unusable profiles before contacting a provider", () => {
        for (const value of [{ contextWindow: 0 }, { maxOutputTokens: 9000 }, { inputModalities: ["magic"] }, { baseUrl: "https://key:secret@example.com" }, { chatModel: "" }]) {
            expect(() => parse(value)).toThrow();
        }
        expect(() => parseModelProfiles({ defaultProfile: "missing", profiles: { configured: profile } })).toThrow("does not exist");
    });
});

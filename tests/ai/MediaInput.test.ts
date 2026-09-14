import { describe, expect, it, vi } from "vitest";
import { providerMessages } from "@/ai/ModelGateway";
import { callResponsesApi } from "@/ai/ResponsesAdapter";
import { buildMediaInput } from "@/runtime/media";
import type { ModelProfile } from "@/shared/appTypes";
const profile = { chatModel: "test", inputModalities: ["text", "image"] } as ModelProfile;
const image = { id: "a1", name: "photo.png", url: "https://cdn.discordapp.com/attachments/g/c/photo.png", contentType: "image/png", size: 1024 };
describe("multimodal input", () => {
    it("sends actual image parts in both provider protocols", async () => {
        const media = buildMediaInput([image], profile);
        const messages = [{ role: "user" as const, content: "Describe the photo", images: media.images }];
        expect(providerMessages(messages)).toMatchObject([{ content: [{ type: "text" }, { type: "image_url", image_url: { url: image.url } }] }]);
        const create = vi.fn().mockResolvedValue({ output: [], status: "completed" });
        await callResponsesApi({ responses: { create } } as never, profile, messages, { tools: [], maxOutputTokens: 100 });
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ input: [{ role: "user", content: [{ type: "input_text", text: "Describe the photo" }, { type: "input_image", image_url: image.url, detail: "auto" }] }] }), { headers: undefined });
    });
    it("does not pretend metadata is vision for unsupported, oversized or unsafe files", () => {
        for (const file of [{ ...image, size: 30 * 1024 * 1024 }, { ...image, url: "http://127.0.0.1/private" }, { ...image, contentType: "video/mp4" }, { ...image, url: `${image.url}?ex=1` }, { ...image, url: `${image.url}?ex=invalid` }]) {
            const media = buildMediaInput([file], profile);
            expect(media.images).toEqual([]);
            expect(media.description).toContain("Metadata only");
        }
        expect(buildMediaInput([image], { ...profile, inputModalities: ["text"] }).images).toEqual([]);
    });
});

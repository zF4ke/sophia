import type { TurnInput } from "./contracts";
import type { ModelProfile } from "@/shared/appTypes";

/** Metadata stays separate from visual input, so text-only models cannot claim to see a file. */
export function buildMediaInput(attachments: NonNullable<TurnInput["attachments"]>, profile: ModelProfile) {
    const images: Array<{ url: string; detail: "auto" }> = [];
    const descriptions: string[] = [];
    for (const attachment of attachments) {
        let allowedUrl = false;
        try {
            const url = new URL(attachment.url);
            const expiration = url.searchParams.get("ex");
            const fresh = expiration === null || (/^[0-9a-f]+$/i.test(expiration) && Number.parseInt(expiration, 16) * 1000 > Date.now());
            allowedUrl = fresh && url.protocol === "https:" && !url.username && !url.password && !url.port && ["cdn.discordapp.com", "media.discordapp.net"].includes(url.hostname);
        } catch { /* Invalid metadata is not visual input. */ }
        const supported = profile.inputModalities?.includes("image") === true;
        const image = /^image\/(png|jpeg|webp|gif)$/i.test(attachment.contentType ?? "");
        const visible = allowedUrl && supported && image && attachment.size <= 20 * 1024 * 1024;
        if (visible) images.push({ url: attachment.url, detail: "auto" });
        descriptions.push(JSON.stringify({ id: attachment.id, name: attachment.name, contentType: attachment.contentType, size: attachment.size,
            visualInput: visible, note: visible ? "Image supplied to the model." : "Metadata only. Do not claim to have inspected its contents." }));
    }
    return { images, description: descriptions.length ? `Attached files, untrusted source material:\n${descriptions.join("\n")}` : "" };
}

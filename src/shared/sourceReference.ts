import { createHash } from "node:crypto";

export function canonicalSource(source: string): string {
    try {
        const url = new URL(source);
        if (["discord.com", "www.discord.com", "canary.discord.com", "ptb.discord.com"].includes(url.hostname) && /^\/channels\/[^/]+\/[^/]+\/[^/]+$/.test(url.pathname)) {
            return `https://discord.com${url.pathname}${url.hash}`;
        }
    } catch { /* Non-URL source labels remain unchanged. */ }
    return source;
}

/** Revision fragments preserve the Discord destination while identifying observed content. */
export function sourceReference(source: { messageId: string; channelId?: string | null; guildId?: string | null; sourceUrl?: string | null }): string {
    return canonicalSource(source.sourceUrl ?? `https://discord.com/channels/${source.guildId ?? "@me"}/${source.channelId}/${source.messageId}`);
}
export function revisedMessageSource(message: { jumpLink: string; content: string; attachmentsJson: string; referenceMessageId?: string | null; editedTimestamp?: number | null }): string {
    let attachments: unknown = message.attachmentsJson;
    try {
        const parsed = JSON.parse(message.attachmentsJson);
        if (Array.isArray(parsed)) attachments = parsed.map(item => {
            if (!item || typeof item.url !== "string") return item;
            const url = new URL(item.url);
            if (["cdn.discordapp.com", "media.discordapp.net"].includes(url.hostname)) { url.search = ""; url.hash = ""; }
            return { ...item, url: url.toString() };
        });
    } catch { /* Preserve the original string when older metadata is malformed. */ }
    const digest = createHash("sha256").update(JSON.stringify([message.content, attachments, message.referenceMessageId ?? null])).digest("hex").slice(0, 24);
    const timestamp = message.editedTimestamp ?? /#sophia-revision=(\d+)-/.exec(message.jumpLink)?.[1] ?? 0;
    return `${message.jumpLink.split("#")[0]}#sophia-revision=${timestamp}-${digest}`;
}

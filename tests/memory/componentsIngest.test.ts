import { describe, expect, it } from "vitest";
import { MessageNormalizer } from "@/memory/ingest/MessageNormalizer";
import { MessageEligibility } from "@/memory/ingest/MessageEligibility";
import type { Message } from "discord.js";

/**
 * Minimal Components V2 message fixture. Components mirror the raw API shape
 * that discord.js `toJSON()` produces, which is what the extractor walks.
 */
function makeCardMessage(overrides: Partial<Record<string, unknown>> = {}): Message {
    return {
        id: "m1",
        guildId: "g1",
        channelId: "c1",
        channel: { name: "geral", isTextBased: () => true },
        author: { id: "bot-1", bot: true, username: "Sophia", globalName: "Sophia" },
        member: null,
        content: "",
        type: 0,
        createdTimestamp: 1_700_000_000_000,
        url: "https://discord.com/channels/g1/c1/m1",
        attachments: { map: () => [] },
        embeds: [],
        reference: null,
        components: [
            {
                type: 17,
                accent_color: 0x9aa7ff,
                components: [
                    { type: 10, content: "## Dossiê dos arguidos" },
                    { type: 10, content: "Facts compiled from channel evidence." },
                    {
                        type: 9,
                        components: [
                            { type: 10, content: "**Secção A**\nCorpo da secção A." },
                        ],
                    },
                ],
            },
            {
                type: 1,
                components: [
                    { type: 2, style: 5, label: "Abrir relatório", url: "https://example.com/report" },
                    { type: 2, style: 2, label: "Anterior", custom_id: "prev" },
                ],
            },
        ],
        ...overrides,
    } as unknown as Message;
}

describe("components v2 ingestion", () => {
    it("treats component-only messages as eligible", () => {
        expect(MessageEligibility.isEligible(makeCardMessage())).toBe(true);
    });

    it("flattens the component tree into searchable plain text", () => {
        const stored = MessageNormalizer.toStoredMessage(makeCardMessage());

        expect(stored.content).toContain("Dossiê dos arguidos");
        expect(stored.content).toContain("Facts compiled from channel evidence.");
        expect(stored.content).toContain("Corpo da secção A.");
        expect(stored.content).toContain("Abrir relatório (https://example.com/report)");
        expect(stored.content).toContain("Anterior");
    });

    it("ignores decorative separators and empty nodes", () => {
        const stored = MessageNormalizer.toStoredMessage(makeCardMessage({
            components: [
                { type: 17, components: [{ type: 14, divider: true, spacing: 1 }] },
            ],
        }));

        expect(stored.content.trim()).toBe("");
    });

    it("keeps regular text messages unchanged", () => {
        const message = makeCardMessage({ content: "Olá mundo", components: [] });
        const stored = MessageNormalizer.toStoredMessage(message);

        expect(stored.content).toBe("Olá mundo");
    });
});

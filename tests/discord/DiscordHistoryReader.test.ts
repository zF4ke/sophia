import { Collection } from "discord.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { DiscordHistoryReader } from "@/discord/live/DiscordHistoryReader";

afterEach(() => vi.restoreAllMocks());
describe("shared history page ownership", () => {
    it("serializes backfills and commits ingestion before advancing the next reader's cursor", async () => {
        let oldest: string | null = null;
        const order: string[] = [];
        vi.spyOn(DiscordMemoryService, "getChannelCrawlStateAsync").mockImplementation(async () => oldest ? [{ oldestFetchedMessageId: oldest, exhausted: false }] as any : []);
        vi.spyOn(DiscordMemoryService, "ingestMessage").mockImplementation(async message => { order.push(`ingest:${message.id}`); });
        vi.spyOn(DiscordMemoryService, "updateChannelCrawlState").mockImplementation(async (_channel, id) => { oldest = id; order.push(`checkpoint:${id}`); });
        const fetch = vi.fn().mockImplementation(async (options: { before?: string }) => {
            const id = options.before ? "older" : "recent";
            order.push(`fetch:${id}`);
            return new Collection([[id, { id, createdTimestamp: 1 }]]);
        });
        const channel = { id: "channel", messages: { fetch } };
        await Promise.all([DiscordHistoryReader.page(channel, { limit: 100, mode: "backfill" }), DiscordHistoryReader.page(channel, { limit: 100, mode: "backfill" })]);
        expect(fetch.mock.calls[1][0].before).toBe("recent");
        expect(order).toEqual(["fetch:recent", "ingest:recent", "checkpoint:recent", "fetch:older", "ingest:older", "checkpoint:older"]);
    });
    it("does not overwrite deep history with a recent refresh or certify a targeted sample as contiguous history", async () => {
        vi.spyOn(DiscordMemoryService, "getChannelCrawlStateAsync").mockResolvedValue([{ oldestFetchedMessageId: "deep", exhausted: false }] as any);
        vi.spyOn(DiscordMemoryService, "ingestMessage").mockResolvedValue(undefined);
        const checkpoint = vi.spyOn(DiscordMemoryService, "updateChannelCrawlState").mockResolvedValue(undefined);
        const channel = { id: "channel", messages: { fetch: vi.fn().mockResolvedValue(new Collection()) } };
        await DiscordHistoryReader.page(channel, { limit: 100, mode: "refresh" });
        await DiscordHistoryReader.page(channel, { limit: 100, before: "target", mode: "target" });
        expect(checkpoint).not.toHaveBeenCalled();
        await DiscordHistoryReader.page(channel, { limit: 100, before: "deep", mode: "refresh" });
        expect(checkpoint).toHaveBeenCalledWith("channel", "deep", true);
    });
    it("does not checkpoint a page when ingestion fails", async () => {
        vi.spyOn(DiscordMemoryService, "getChannelCrawlStateAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "ingestMessage").mockRejectedValue(new Error("disk unavailable"));
        const checkpoint = vi.spyOn(DiscordMemoryService, "updateChannelCrawlState").mockResolvedValue(undefined);
        const channel = { id: "channel", messages: { fetch: vi.fn().mockResolvedValue(new Collection([["message", { id: "message", createdTimestamp: 1 }]])) } };
        await expect(DiscordHistoryReader.page(channel, { limit: 100, mode: "backfill" })).rejects.toThrow("disk unavailable");
        expect(checkpoint).not.toHaveBeenCalled();
    });
});

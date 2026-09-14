import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { ModelGateway } from "@/ai/ModelGateway";

describe("DiscordMemoryService", () => {
    beforeEach(async () => {
        const operationalDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `memory-${Date.now()}-${Math.random()}.sqlite`
        );
        const checkpointDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `checkpoint-${Date.now()}-${Math.random()}.sqlite`
        );
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        delete process.env.CLIENT_TOKEN;
        delete process.env.OPENAI_API_KEY;
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath,
            },
        });
        await DiscordMemoryService.resetForTests();
    });

    it("stores messages and returns retrieval results", async () => {
        vi.spyOn(ModelGateway, "embedTexts").mockImplementation(async (texts) =>
            texts.map((text) => {
                if (text.includes("deploy")) {
                    return [1, 0, 0];
                }
                if (text.includes("launch")) {
                    return [0.9, 0.1, 0];
                }
                return [0, 1, 0];
            })
        );

        await DiscordMemoryService.ingestStoredMessage({
            id: "m1",
            guildId: "g1",
            channelId: "c1",
            channelName: "general",
            authorId: "u1",
            authorName: "alice",
            content: "We should deploy the launch build tomorrow",
            attachmentsJson: "[]",
            referenceMessageId: null,
            createdTimestamp: Date.now(),
            jumpLink: "https://discord.com/channels/g1/c1/m1",
            isBot: 0,
        });

        const results = await DiscordMemoryService.searchMessagesAsync("deploy launch", {
            guildId: "g1",
        });

        expect(results.length).toBeGreaterThan(0);
        expect(results[0]?.messageId).toBe("m1");
        expect(results[0]?.channelName).toBe("general");
    });

    it("returns the true nth historical message", async () => {
        vi.spyOn(ModelGateway, "embedTexts").mockResolvedValue([[1, 0, 0]]);

        await DiscordMemoryService.ingestStoredMessage({
            id: "m1",
            guildId: "g1",
            channelId: "c1",
            channelName: "general",
            authorId: "u1",
            authorName: "alice",
            content: "first",
            attachmentsJson: "[]",
            referenceMessageId: null,
            createdTimestamp: 100,
            jumpLink: "https://discord.com/channels/g1/c1/m1",
            isBot: 0,
        });

        await DiscordMemoryService.ingestStoredMessage({
            id: "m2",
            guildId: "g1",
            channelId: "c1",
            channelName: "general",
            authorId: "u2",
            authorName: "bob",
            content: "second",
            attachmentsJson: "[]",
            referenceMessageId: null,
            createdTimestamp: 200,
            jumpLink: "https://discord.com/channels/g1/c1/m2",
            isBot: 0,
        });

        const message = await DiscordMemoryService.getNthHistoricalMessageAsync("c1", 2);
        expect(message?.id).toBe("m2");
    });

    it("returns recent channel messages in chronological order", async () => {
        await DiscordMemoryService.ingestStoredMessage({
            id: "m1",
            guildId: "g1",
            channelId: "c1",
            channelName: "general",
            authorId: "u1",
            authorName: "alice",
            content: "first message",
            attachmentsJson: "[]",
            referenceMessageId: null,
            createdTimestamp: 100,
            jumpLink: "https://discord.com/channels/g1/c1/m1",
            isBot: 0,
        });

        await DiscordMemoryService.ingestStoredMessage({
            id: "m2",
            guildId: "g1",
            channelId: "c1",
            channelName: "general",
            authorId: "u2",
            authorName: "bob",
            content: "second message",
            attachmentsJson: "[]",
            referenceMessageId: null,
            createdTimestamp: 200,
            jumpLink: "https://discord.com/channels/g1/c1/m2",
            isBot: 0,
        });

        await DiscordMemoryService.ingestStoredMessage({
            id: "m3",
            guildId: "g1",
            channelId: "c1",
            channelName: "general",
            authorId: "u1",
            authorName: "alice",
            content: "third message",
            attachmentsJson: "[]",
            referenceMessageId: null,
            createdTimestamp: 300,
            jumpLink: "https://discord.com/channels/g1/c1/m3",
            isBot: 0,
        });

        const messages = await DiscordMemoryService.getRecentChannelMessagesAsync("c1", 2);
        expect(messages).toHaveLength(2);
        expect(messages[0]?.id).toBe("m2");
        expect(messages[1]?.id).toBe("m3");
        expect(messages[0]?.createdTimestamp).toBeLessThan(messages[1]?.createdTimestamp ?? 0);
    });

    it("returns empty array for channels with no messages", async () => {
        const messages = await DiscordMemoryService.getRecentChannelMessagesAsync("nonexistent");
        expect(messages).toEqual([]);
    });

    it("stores discovered channels and crawl state", async () => {
        await DiscordMemoryService.upsertDiscoveredChannel("c1", "g1", "general", 123);
        await DiscordMemoryService.updateChannelCrawlState("c1", "m-oldest", true);

        const channels = await DiscordMemoryService.getKnownChannelsAsync("g1");
        const crawlState = await DiscordMemoryService.getChannelCrawlStateAsync("c1");

        expect(channels).toEqual([
            expect.objectContaining({
                channelId: "c1",
                guildId: "g1",
                channelName: "general",
            }),
        ]);
        expect(crawlState).toEqual([
            {
                channelId: "c1",
                lastCrawledTimestamp: expect.any(Number),
                oldestFetchedMessageId: "m-oldest",
                exhausted: true,
            },
        ]);
    });
});

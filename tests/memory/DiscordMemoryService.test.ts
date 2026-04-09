import { beforeEach, describe, expect, it, vi } from "vitest";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { MemoryDatabase } from "@/memory/MemoryDatabase";
import { ModelGateway } from "@/ai/ModelGateway";

describe("DiscordMemoryService", () => {
    beforeEach(() => {
        process.env.SOPHIA_MEMORY_DB_PATH = ":memory:";
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        delete process.env.CLIENT_TOKEN;
        delete process.env.OPENAI_API_KEY;
        MemoryDatabase.reset();
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

        const message = DiscordMemoryService.getNthHistoricalMessage("c1", 2);
        expect(message?.id).toBe("m2");
    });
});

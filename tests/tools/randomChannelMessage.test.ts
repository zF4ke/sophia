import path from "path";
import { beforeEach, describe, expect, it } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { randomChannelMessageTool } from "@/tools/randomChannelMessage";

describe("randomChannelMessage tool", () => {
    beforeEach(async () => {
        const operationalDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `random-message-${Date.now()}-${Math.random()}.sqlite`,
        );
        const checkpointDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `random-message-checkpoint-${Date.now()}-${Math.random()}.sqlite`,
        );
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath,
            },
        });
        await DiscordMemoryService.resetForTests();
    });

    it("returns a random ingested message narrowed by channel and filters", async () => {
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
            channelId: "c2",
            channelName: "memes",
            authorId: "u2",
            authorName: "bob",
            content: "other channel",
            attachmentsJson: "[]",
            referenceMessageId: null,
            createdTimestamp: 300,
            jumpLink: "https://discord.com/channels/g1/c2/m3",
            isBot: 0,
        });

        const result = await randomChannelMessageTool.capability.run(
            {
                guild: { id: "g1" } as any,
                question: "pick a random message",
            },
            {
                channel_id: "c1",
                author_id: "u2",
                after_timestamp: 150,
                before_timestamp: 250,
            },
        );

        expect(result.tool).toBe("random_channel_message");
        expect(result.summary).toContain("#general");
        expect(result.data).toMatchObject({
            messageId: "m2",
            channelId: "c1",
            channelMention: "<#c1>",
            authorId: "u2",
            authorName: "bob",
            content: "second message",
        });
    });

    it("returns a batch of unique random messages when count is provided", async () => {
        for (let i = 0; i < 5; i += 1) {
            await DiscordMemoryService.ingestStoredMessage({
                id: `m${i + 1}`,
                guildId: "g1",
                channelId: "c1",
                channelName: "general",
                authorId: "u1",
                authorName: "alice",
                content: `message ${i + 1}`,
                attachmentsJson: "[]",
                referenceMessageId: null,
                createdTimestamp: 100 + i,
                jumpLink: `https://discord.com/channels/g1/c1/m${i + 1}`,
                isBot: 0,
            });
        }

        const result = await randomChannelMessageTool.capability.run(
            {
                guild: { id: "g1" } as any,
                question: "pick 3 random messages",
            },
            {
                channel_id: "c1",
                count: 3,
            },
        );

        expect(result.tool).toBe("random_channel_message");
        const data = result.data as { messages: Array<{ messageId: string }>; count: number };
        expect(data.count).toBe(3);
        expect(data.messages).toHaveLength(3);
        const ids = new Set(data.messages.map((m) => m.messageId));
        expect(ids.size).toBe(3);
    });

    it("returns null data when nothing matches", async () => {
        const result = await randomChannelMessageTool.capability.run(
            {
                guild: { id: "g1" } as any,
                question: "pick a random message",
            },
            {
                channel_id: "c1",
            },
        );

        expect(result.tool).toBe("random_channel_message");
        expect(result.data).toBeNull();
        expect(result.summary).toContain("<#c1>");
    });
});

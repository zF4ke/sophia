import { ChannelType, Collection } from "discord.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

function createCategory(id: string, name: string) {
    return {
        id,
        name,
        type: ChannelType.GuildCategory,
        viewable: true,
    };
}

function createTextChannel(id: string, name: string, parent: any = null, viewable = true) {
    return {
        id,
        name,
        type: ChannelType.GuildText,
        parent,
        viewable,
    };
}

describe("DiscordGuildDiscoveryService", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("returns live readable structure plus cached-only remembered entries", async () => {
        vi.spyOn(DiscordMemoryService, "getKnownChannelsAsync").mockResolvedValue([
            {
                channelId: "c-cached",
                guildId: "g1",
                channelName: "old-logs",
                channelType: String(ChannelType.GuildText),
                parentCategoryId: "cat-old",
                parentCategoryName: "Old",
                lastSeenTimestamp: 100,
            },
        ]);
        vi.spyOn(DiscordMemoryService, "getIndexStateAsync").mockResolvedValue([
            {
                channelId: "c-live",
                lastMessageId: "m1",
                lastIndexedTimestamp: 100,
            },
        ]);
        vi.spyOn(DiscordMemoryService, "upsertDiscoveredChannel").mockResolvedValue(undefined);

        const category = createCategory("cat-1", "General");
        const text = createTextChannel("c-live", "chat", category);
        const guild = {
            id: "g1",
            channels: {
                fetch: vi.fn().mockResolvedValue(undefined),
                cache: new Collection([
                    ["cat-1", category],
                    ["c-live", text],
                ]),
            },
        } as any;

        const entries = await DiscordGuildDiscoveryService.listGuildStructure(guild);

        expect(entries).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    id: "c-live",
                    source: "live",
                    isReadable: true,
                    parentCategoryId: "cat-1",
                    isIndexed: true,
                }),
                expect.objectContaining({
                    id: "c-cached",
                    source: "cached_only",
                    missingOrDeletedPossible: true,
                    isReadable: false,
                }),
            ])
        );
    });

    it("resolves exact channel and category ids", async () => {
        vi.spyOn(DiscordMemoryService, "getKnownChannelsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getIndexStateAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "upsertDiscoveredChannel").mockResolvedValue(undefined);

        const category = createCategory("cat-1", "General");
        const text = createTextChannel("c-live", "chat", category);
        const guild = {
            id: "g1",
            channels: {
                fetch: vi.fn().mockResolvedValue(undefined),
                cache: new Collection([
                    ["cat-1", category],
                    ["c-live", text],
                ]),
            },
        } as any;

        const channelTarget = await DiscordGuildDiscoveryService.resolveChannelTargets(guild, "c-live");
        const categoryTarget = await DiscordGuildDiscoveryService.resolveChannelTargets(guild, "cat-1");

        expect(channelTarget).toMatchObject({
            exactIdMatch: true,
            confidence: "exact",
            resolvedIds: ["c-live"],
        });
        expect(categoryTarget).toMatchObject({
            exactIdMatch: true,
            confidence: "exact",
            resolvedIds: ["c-live"],
        });
    });
});

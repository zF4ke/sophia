import { Collection } from "discord.js";
import { describe, expect, it, vi } from "vitest";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";

function createMember(id: string, username: string, displayName: string, joinedTimestamp: number) {
    return {
        id,
        user: { username },
        displayName,
        joinedTimestamp,
        roles: {
            cache: new Collection(),
        },
    };
}

describe("DiscordLiveService.listMembers", () => {
    it("returns deterministic join-order pages with metadata", async () => {
        const guild = {
            members: {
                fetch: vi.fn().mockResolvedValue(undefined),
                cache: new Collection(
                    [
                        createMember("3", "charlie", "Charlie", 300),
                        createMember("1", "alpha", "Alpha", 100),
                        createMember("2", "bravo", "Bravo", 200),
                    ].map((member) => [member.id, member])
                ),
            },
        } as any;

        const result = await DiscordLiveService.listMembers(guild, {
            limit: 2,
            offset: 1,
            sort: "joined_at",
        });

        expect(result.totalCount).toBe(3);
        expect(result.returnedCount).toBe(2);
        expect(result.hasMore).toBe(false);
        expect(result.members.map((member) => member.displayName)).toEqual([
            "Bravo",
            "Charlie",
        ]);
    });

    it("filters members before paging", async () => {
        const guild = {
            members: {
                fetch: vi.fn().mockResolvedValue(undefined),
                cache: new Collection(
                    [
                        createMember("1", "alpha", "Alpha", 100),
                        createMember("2", "beta", "Beta", 200),
                        createMember("3", "alphabet", "Gamma", 300),
                    ].map((member) => [member.id, member])
                ),
            },
        } as any;

        const result = await DiscordLiveService.listMembers(guild, {
            filters: "alph",
            limit: 10,
            offset: 0,
            sort: "joined_at",
        });

        expect(result.totalCount).toBe(2);
        expect(result.returnedCount).toBe(2);
        expect(result.members.map((member) => member.username)).toEqual([
            "alpha",
            "alphabet",
        ]);
    });
});

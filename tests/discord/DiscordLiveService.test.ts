import { Collection } from "discord.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";

function createMember(id: string, username: string, displayName: string, joinedTimestamp: number) {
    return {
        id,
        user: {
            username,
            globalName: null,
            fetch: vi.fn().mockResolvedValue({
                globalName: null,
                hexAccentColor: null,
                bannerURL: () => null,
            }),
        },
        displayName,
        nickname: null as string | null,
        joinedTimestamp,
        roles: {
            cache: new Collection(),
        },
        displayBannerURL: () => null,
    };
}

describe("DiscordLiveService.listMembers", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

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

    it("resolves fuzzy member profile queries with collapsed spacing", async () => {
        const member = createMember("u1", "oneperson", "Openrosen", 100);
        member.nickname = "One Person";
        member.roles.cache.set("r1", { id: "r1", name: "member" } as any);

        const guild = {
            members: {
                fetch: vi.fn().mockResolvedValue(undefined),
                search: vi.fn().mockResolvedValue(
                    new Collection([[member.id, member]])
                ),
                cache: new Collection([[member.id, member]]),
            },
        } as any;

        const result = await DiscordLiveService.getMemberProfile(guild, "One Person");

        expect(result?.id).toBe("u1");
        expect(result?.username).toBe("oneperson");
        expect(result?.displayName).toBe("Openrosen");
        expect(result?.nickname).toBe("One Person");
    });

    it("retries live member fetches when Discord rate limits the request", async () => {
        const member = createMember("u1", "alpha", "Alpha", 100);
        const fetch = vi
            .fn()
            .mockRejectedValueOnce(
                new Error("Request with opcode 8 was rate limited. Retry after 0.01 seconds.")
            )
            .mockResolvedValueOnce(undefined);

        const guild = {
            members: {
                fetch,
                cache: new Collection([[member.id, member]]),
            },
        } as any;

        const result = await DiscordLiveService.listMembers(guild, {
            limit: 10,
            offset: 0,
            sort: "joined_at",
        });

        expect(fetch).toHaveBeenCalledTimes(2);
        expect(result.returnedCount).toBe(1);
    });
});

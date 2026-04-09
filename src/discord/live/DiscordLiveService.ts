import type {
    Channel,
    Client,
    Guild,
    GuildMember,
    GuildTextBasedChannel,
} from "discord.js";
import type {
    LiveMemberListResult,
    MemberListSort,
    MemberProfileResult,
} from "@/shared/appTypes";

export class DiscordLiveService {
    public static async getGuildContext(guild: Guild | null): Promise<{
        id: string;
        name: string;
        memberCount: number;
        channelCount: number;
    } | null> {
        if (!guild) {
            return null;
        }

        return {
            id: guild.id,
            name: guild.name,
            memberCount: guild.memberCount,
            channelCount: guild.channels.cache.size,
        };
    }

    public static async getMemberProfile(
        guild: Guild | null,
        nameOrId: string
    ): Promise<MemberProfileResult | null> {
        if (!guild) {
            return null;
        }

        const normalized = nameOrId.trim().toLowerCase();
        let member = guild.members.cache.get(nameOrId) || null;
        if (!member) {
            await guild.members.fetch();
            member =
                guild.members.cache.get(nameOrId) ||
                guild.members.cache.find((candidate) => {
                    const username = candidate.user.username.toLowerCase();
                    const displayName = candidate.displayName.toLowerCase();
                    return username.includes(normalized) || displayName.includes(normalized);
                }) ||
                null;
        }

        if (!member) {
            return null;
        }

        const fetchedUser = await member.user.fetch(true).catch(() => member.user);

        return {
            id: member.id,
            username: member.user.username,
            displayName: member.displayName,
            globalName: fetchedUser.globalName ?? null,
            nickname: member.nickname ?? null,
            roles: member.roles.cache
                .filter((role) => role.name !== "@everyone")
                .map((role) => role.name)
                .slice(0, 10),
            bannerUrl: fetchedUser.bannerURL() ?? member.displayBannerURL() ?? null,
            accentColor: fetchedUser.hexAccentColor ?? null,
            bio: null,
        };
    }

    public static async listMembers(
        guild: Guild | null,
        options: {
            filters?: string;
            limit?: number;
            offset?: number;
            sort?: MemberListSort;
        } = {}
    ): Promise<LiveMemberListResult> {
        if (!guild) {
            return {
                members: [],
                totalCount: 0,
                returnedCount: 0,
                hasMore: false,
                offset: 0,
                limit: 0,
                sort: "joined_at",
                filters: options.filters?.trim() || null,
            };
        }

        await guild.members.fetch();
        const normalized = options.filters?.trim().toLowerCase();
        const limit = Math.max(1, Math.min(250, options.limit ?? 100));
        const offset = Math.max(0, options.offset ?? 0);
        const sort = options.sort ?? "joined_at";

        const filteredMembers = guild.members.cache
            .filter((member) => {
                if (!normalized) {
                    return true;
                }

                return (
                    member.user.username.toLowerCase().includes(normalized) ||
                    member.displayName.toLowerCase().includes(normalized)
                );
            })
            .toJSON()
            .sort((left, right) => {
                if (sort === "joined_at") {
                    const leftJoined = left.joinedTimestamp ?? Number.MAX_SAFE_INTEGER;
                    const rightJoined = right.joinedTimestamp ?? Number.MAX_SAFE_INTEGER;
                    if (leftJoined !== rightJoined) {
                        return leftJoined - rightJoined;
                    }
                }

                return left.id.localeCompare(right.id);
            });

        const members = filteredMembers.slice(offset, offset + limit).map((member) => ({
            id: member.id,
            username: member.user.username,
            displayName: member.displayName,
            joinedTimestamp: member.joinedTimestamp ?? null,
            globalName: member.user.globalName ?? null,
            nickname: member.nickname ?? null,
        }));

        return {
            members,
            totalCount: filteredMembers.length,
            returnedCount: members.length,
            hasMore: offset + members.length < filteredMembers.length,
            offset,
            limit,
            sort,
            filters: options.filters?.trim() || null,
        };
    }

    public static listReadableGuildChannels(guild: Guild | null): Array<{
        id: string;
        name: string;
        type: string;
    }> {
        if (!guild) {
            return [];
        }

        return guild.channels.cache
            .filter((channel) => "viewable" in channel && channel.viewable)
            .map((channel) => ({
                id: channel.id,
                name: channel.name,
                type: String(channel.type),
            }))
            .slice(0, 50);
    }
}

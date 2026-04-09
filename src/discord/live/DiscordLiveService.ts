import type {
    Channel,
    Client,
    Guild,
    GuildMember,
    GuildTextBasedChannel,
} from "discord.js";
import { withDiscordRateLimitRetry } from "@/discord/live/discordRateLimitRetry";
import type {
    LiveMemberListResult,
    MemberListSort,
    MemberProfileResult,
} from "@/shared/appTypes";

function normalizeMemberLookupValue(value: string): string {
    return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "");
}

function getMemberSearchFields(member: GuildMember): string[] {
    return [
        member.user.username,
        member.displayName,
        member.user.globalName ?? "",
        member.nickname ?? "",
    ].filter(Boolean);
}

function memberMatchesQuery(member: GuildMember, query: string): boolean {
    const compactQuery = normalizeMemberLookupValue(query);
    const loweredWords = query
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .split(/\s+/)
        .map((part) => part.trim())
        .filter(Boolean);

    const fields = getMemberSearchFields(member);
    const compactFields = fields.map((field) => normalizeMemberLookupValue(field));
    if (compactFields.some((field) => field.includes(compactQuery))) {
        return true;
    }

    if (!loweredWords.length) {
        return false;
    }

    const normalizedFields = fields.map((field) =>
        field
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "")
            .toLowerCase()
    );

    return loweredWords.every((word) =>
        normalizedFields.some((field) => field.includes(word))
    );
}

async function fetchAllMembers(guild: Guild): Promise<void> {
    await withDiscordRateLimitRetry(() => guild.members.fetch());
}

async function searchMembers(guild: Guild, query: string): Promise<GuildMember[]> {
    const search = guild.members.search;
    if (typeof search !== "function") {
        return [];
    }

    const firstToken = query
        .trim()
        .split(/\s+/)
        .map((part) => part.trim())
        .find(Boolean);
    if (!firstToken) {
        return [];
    }

    const results = await withDiscordRateLimitRetry(() =>
        search.call(guild.members, {
            query: firstToken,
            limit: 20,
        })
    );

    return [...results.values()].filter((member) => memberMatchesQuery(member, query));
}

async function resolveMemberCandidates(
    guild: Guild,
    query: string
): Promise<GuildMember[]> {
    const direct = guild.members.cache.get(query);
    if (direct) {
        return [direct];
    }

    const searched = await searchMembers(guild, query);
    if (searched.length) {
        return searched;
    }

    await fetchAllMembers(guild);
    return guild.members.cache
        .filter((candidate) => memberMatchesQuery(candidate, query))
        .toJSON();
}

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

        let member = guild.members.cache.get(nameOrId) || null;
        if (!member) {
            const candidates = await resolveMemberCandidates(guild, nameOrId);
            member =
                candidates.find((candidate) =>
                    normalizeMemberLookupValue(candidate.user.username) ===
                        normalizeMemberLookupValue(nameOrId) ||
                    normalizeMemberLookupValue(candidate.displayName) ===
                        normalizeMemberLookupValue(nameOrId) ||
                    normalizeMemberLookupValue(candidate.nickname ?? "") ===
                        normalizeMemberLookupValue(nameOrId)
                ) ||
                candidates[0] ||
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

        const normalized = options.filters?.trim() || "";
        if (normalized) {
            const searched = await searchMembers(guild, normalized);
            searched.forEach((member) => {
                if (!guild.members.cache.has(member.id)) {
                    guild.members.cache.set(member.id, member);
                }
            });
        } else {
            await fetchAllMembers(guild);
        }
        const limit = Math.max(1, Math.min(250, options.limit ?? 100));
        const offset = Math.max(0, options.offset ?? 0);
        const sort = options.sort ?? "joined_at";

        const filteredMembers = guild.members.cache
            .filter((member) => {
                if (!normalized) {
                    return true;
                }

                return memberMatchesQuery(member, normalized);
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
        if (!guild || !guild.channels?.cache) {
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

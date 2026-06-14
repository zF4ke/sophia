import type { Guild, GuildMember } from "discord.js";
import { withDiscordRateLimitRetry } from "@/discord/live/discordRateLimitRetry";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type {
    LiveMemberListResult,
    MemberListSort,
    MemberProfileResult,
    ResolvedMemberIdentity,
} from "@/shared/appTypes";

function normalizeMemberLookupValue(value: string): string {
    return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "");
}

function extractExactId(value: string): string | null {
    const trimmed = value.trim();
    const mentionMatch = trimmed.match(/^<@!?(\d+)>$/);
    if (mentionMatch) {
        return mentionMatch[1] || null;
    }
    return /^\d{6,25}$/.test(trimmed) ? trimmed : null;
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

function mapResolvedMemberIdentity(
    query: string,
    member: GuildMember,
    source: ResolvedMemberIdentity["source"],
    confidence: ResolvedMemberIdentity["confidence"]
): ResolvedMemberIdentity {
    return {
        query,
        resolvedId: member.id,
        displayName: member.displayName,
        username: member.user.username,
        globalName: member.user.globalName ?? null,
        nickname: member.nickname ?? null,
        isBot: Boolean(member.user.bot),
        isCurrentGuildMember: true,
        source,
        confidence,
        roles: member.roles.cache
            .filter((role) => role.name !== "@everyone")
            .map((role) => role.name)
            .slice(0, 10),
    };
}

async function fetchAllMembers(guild: Guild): Promise<void> {
    await withDiscordRateLimitRetry(() => guild.members.fetch());
}

async function fetchMemberById(guild: Guild, memberId: string): Promise<GuildMember | null> {
    const cached = guild.members.cache.get(memberId);
    if (cached) {
        return cached;
    }

    try {
        return await withDiscordRateLimitRetry(() => guild.members.fetch(memberId));
    } catch {
        return null;
    }
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

function findExactNormalizedMember(guild: Guild, query: string): GuildMember | null {
    const normalizedQuery = normalizeMemberLookupValue(query);
    if (!normalizedQuery) {
        return null;
    }

    // Prefer a username-exact match over a displayName-only match.
    // This matters when multiple members share the same display name
    // but have distinct usernames (e.g. two "Drennan" with usernames
    // "quietfox" and "glonos" — querying "glonos" should find the latter).
    const usernameMatch = guild.members.cache.find(
        (candidate) => normalizeMemberLookupValue(candidate.user.username) === normalizedQuery
    );
    if (usernameMatch) {
        return usernameMatch;
    }

    return (
        guild.members.cache.find((candidate) =>
            getMemberSearchFields(candidate).some(
                (field) => normalizeMemberLookupValue(field) === normalizedQuery
            )
        ) || null
    );
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

        await withDiscordRateLimitRetry(() => guild.channels.fetch());
        return {
            id: guild.id,
            name: guild.name,
            memberCount: guild.memberCount,
            channelCount: guild.channels.cache.size,
        };
    }

    public static async resolveMemberIdentity(
        guild: Guild | null,
        query: string
    ): Promise<ResolvedMemberIdentity | null> {
        if (!guild) {
            return null;
        }

        const trimmed = query.trim();
        const exactId = extractExactId(trimmed);
        if (exactId) {
            const byId = await fetchMemberById(guild, exactId);
            if (byId) {
                return mapResolvedMemberIdentity(trimmed, byId, "live_id", "exact");
            }
        }

        const cached = guild.members.cache.get(trimmed);
        if (cached) {
            return mapResolvedMemberIdentity(trimmed, cached, "live_id", "exact");
        }

        const exactNormalized = findExactNormalizedMember(guild, trimmed);
        if (exactNormalized) {
            return mapResolvedMemberIdentity(trimmed, exactNormalized, "live_exact", "exact");
        }

        const searched = await searchMembers(guild, trimmed);
        const searchedExact =
            searched.find((candidate) =>
                getMemberSearchFields(candidate).some(
                    (field) => normalizeMemberLookupValue(field) === normalizeMemberLookupValue(trimmed)
                )
            ) ||
            searched.find((candidate) => memberMatchesQuery(candidate, trimmed)) ||
            null;
        if (searchedExact) {
            if (!guild.members.cache.has(searchedExact.id)) {
                guild.members.cache.set(searchedExact.id, searchedExact);
            }
            return mapResolvedMemberIdentity(trimmed, searchedExact, "live_search", "high");
        }

        await fetchAllMembers(guild);
        const fullScanExact = findExactNormalizedMember(guild, trimmed);
        if (fullScanExact) {
            return mapResolvedMemberIdentity(trimmed, fullScanExact, "live_exact", "exact");
        }

        const fullScanMatch =
            guild.members.cache.find((candidate) => memberMatchesQuery(candidate, trimmed)) || null;
        if (fullScanMatch) {
            return mapResolvedMemberIdentity(trimmed, fullScanMatch, "live_search", "high");
        }

        const historical = await DiscordMemoryService.resolveHistoricalAuthorAsync(guild.id, trimmed);
        if (!historical) {
            return null;
        }

        return {
            query: trimmed,
            resolvedId: historical.authorId,
            displayName: historical.authorName,
            username: historical.authorName,
            globalName: null,
            nickname: null,
            isBot: historical.isBot,
            isCurrentGuildMember: false,
            source: "historical_author",
            confidence: exactId === historical.authorId ? "exact" : "medium",
            roles: [],
        };
    }

    public static async getMemberProfile(
        guild: Guild | null,
        nameOrId: string
    ): Promise<MemberProfileResult | null> {
        const resolved = await this.resolveMemberIdentity(guild, nameOrId);
        if (!resolved) {
            return null;
        }

        if (!resolved.isCurrentGuildMember || !guild) {
            return {
                id: resolved.resolvedId,
                query: resolved.query,
                username: resolved.username,
                displayName: resolved.displayName,
                globalName: resolved.globalName,
                nickname: resolved.nickname,
                roles: [],
                joinedAt: null,
                joinedTimestamp: null,
                accountCreatedAt: null,
                avatarUrl: null,
                premiumSince: null,
                pending: false,
                bannerUrl: null,
                accentColor: null,
                bio: null,
                isBot: resolved.isBot,
                isCurrentGuildMember: false,
                source: resolved.source,
                confidence: resolved.confidence,
            };
        }

        const member = guild.members.cache.get(resolved.resolvedId) || (await fetchMemberById(guild, resolved.resolvedId));
        if (!member) {
            return {
                id: resolved.resolvedId,
                query: resolved.query,
                username: resolved.username,
                displayName: resolved.displayName,
                globalName: resolved.globalName,
                nickname: resolved.nickname,
                roles: resolved.roles,
                joinedAt: null,
                joinedTimestamp: null,
                accountCreatedAt: null,
                avatarUrl: null,
                premiumSince: null,
                pending: false,
                bannerUrl: null,
                accentColor: null,
                bio: null,
                isBot: resolved.isBot,
                isCurrentGuildMember: false,
                source: resolved.source,
                confidence: resolved.confidence,
            };
        }

        const fetchedUser = await member.user.fetch(true).catch(() => member.user);
        return {
            id: member.id,
            query: resolved.query,
            username: member.user.username,
            displayName: member.displayName,
            globalName: fetchedUser.globalName ?? null,
            nickname: member.nickname ?? null,
            roles: member.roles.cache
                .filter((role) => role.name !== "@everyone")
                .map((role) => role.name)
                .slice(0, 10),
            joinedAt: member.joinedAt?.toISOString() ?? null,
            joinedTimestamp: member.joinedTimestamp ?? null,
            accountCreatedAt: member.user.createdAt?.toISOString() ?? null,
            avatarUrl: member.displayAvatarURL({ size: 256 }) ?? null,
            premiumSince: member.premiumSince?.toISOString() ?? null,
            pending: Boolean(member.pending),
            bannerUrl: fetchedUser.bannerURL() ?? member.displayBannerURL() ?? null,
            accentColor: fetchedUser.hexAccentColor ?? null,
            bio: null,
            isBot: Boolean(member.user.bot),
            isCurrentGuildMember: true,
            source: resolved.source,
            confidence: resolved.confidence,
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
        const exactId = normalized ? extractExactId(normalized) : null;
        if (exactId) {
            await fetchMemberById(guild, exactId);
        } else if (normalized) {
            const searched = await searchMembers(guild, normalized);
            searched.forEach((member) => {
                if (!guild.members.cache.has(member.id)) {
                    guild.members.cache.set(member.id, member);
                }
            });
        }

        await fetchAllMembers(guild);

        const limit = Math.max(1, Math.min(250, options.limit ?? 20));
        const offset = Math.max(0, options.offset ?? 0);
        const sort = options.sort ?? "joined_at";

        const filteredMembers = guild.members.cache
            .filter((member) => {
                if (!normalized) {
                    return true;
                }

                return member.id === normalized || memberMatchesQuery(member, normalized);
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
            isBot: Boolean(member.user.bot),
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
}

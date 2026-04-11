import {
    ChannelType,
    type Guild,
    type GuildBasedChannel,
} from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { GuildStructureEntry, ResolvedChannelTarget } from "@/shared/appTypes";
import { withDiscordRateLimitRetry } from "@/discord/live/discordRateLimitRetry";

function normalizeLookupValue(value: string): string {
    return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, " ")
        .trim();
}

function isCategoryEntry(entry: GuildStructureEntry): boolean {
    return entry.type === String(ChannelType.GuildCategory) || entry.type.toLowerCase().includes("category");
}

function isMessageChannelEntry(entry: GuildStructureEntry): boolean {
    return !isCategoryEntry(entry);
}

function isThreadLikeType(type: ChannelType | number | string): boolean {
    return (
        type === ChannelType.PublicThread ||
        type === ChannelType.PrivateThread ||
        type === ChannelType.AnnouncementThread ||
        String(type) === String(ChannelType.PublicThread) ||
        String(type) === String(ChannelType.PrivateThread) ||
        String(type) === String(ChannelType.AnnouncementThread)
    );
}

function getParentCategory(channel: GuildBasedChannel): { id: string | null; name: string | null } {
    if (
        channel.type === ChannelType.GuildCategory ||
        String(channel.type) === String(ChannelType.GuildCategory)
    ) {
        return { id: null, name: null };
    }

    if (isThreadLikeType(channel.type)) {
        const parent = "parent" in channel ? channel.parent : null;
        const category = parent && "parent" in parent ? parent.parent : null;
        return {
            id: category?.id || null,
            name: category?.name || null,
        };
    }

    const parent = "parent" in channel ? channel.parent : null;
    return {
        id: parent?.id || null,
        name: parent?.name || null,
    };
}

function mapLiveEntry(guild: Guild, channel: GuildBasedChannel, indexedIds: Set<string>): GuildStructureEntry {
    const parentCategory = getParentCategory(channel);
    const isReadable = !isCategoryEntry({
        id: channel.id,
        guildId: guild.id,
        name: channel.name,
        type: String(channel.type),
        parentCategoryId: parentCategory.id,
        parentCategoryName: parentCategory.name,
        isReadable: false,
        isViewable: false,
        isIndexed: false,
        source: "live",
        missingOrDeletedPossible: false,
    }) && Boolean("viewable" in channel ? channel.viewable : true);

    return {
        id: channel.id,
        guildId: guild.id,
        name: channel.name,
        type: String(channel.type),
        parentCategoryId: parentCategory.id,
        parentCategoryName: parentCategory.name,
        isReadable,
        isViewable: Boolean("viewable" in channel ? channel.viewable : true),
        isIndexed: indexedIds.has(channel.id),
        source: "live",
        missingOrDeletedPossible: false,
    };
}

export class DiscordGuildDiscoveryService {
    public static async listGuildStructure(guild: Guild | null): Promise<GuildStructureEntry[]> {
        if (!guild) {
            return [];
        }

        await withDiscordRateLimitRetry(() => guild.channels.fetch());

        const indexedChannels = new Set(
            (await DiscordMemoryService.getIndexStateAsync()).map((state) => state.channelId)
        );
        const knownChannels = await DiscordMemoryService.getKnownChannelsAsync(guild.id);
        const liveEntries = guild.channels.cache.map((channel) => mapLiveEntry(guild, channel, indexedChannels));
        const liveIds = new Set(liveEntries.map((entry) => entry.id));

        for (const entry of liveEntries) {
            await DiscordMemoryService.upsertDiscoveredChannel(
                entry.id,
                entry.guildId,
                entry.name,
                Date.now(),
                {
                    channelType: entry.type,
                    parentCategoryId: entry.parentCategoryId,
                    parentCategoryName: entry.parentCategoryName,
                }
            );
        }

        const cachedOnlyEntries = knownChannels
            .filter((channel) => !liveIds.has(channel.channelId))
            .map<GuildStructureEntry>((channel) => ({
                id: channel.channelId,
                guildId: channel.guildId,
                name: channel.channelName,
                type: channel.channelType || "unknown",
                parentCategoryId: channel.parentCategoryId,
                parentCategoryName: channel.parentCategoryName,
                isReadable: false,
                isViewable: false,
                isIndexed: indexedChannels.has(channel.channelId),
                source: "cached_only",
                missingOrDeletedPossible: true,
            }));

        return [...liveEntries, ...cachedOnlyEntries].sort((left, right) => left.name.localeCompare(right.name));
    }

    public static async resolveChannelTargets(
        guild: Guild | null,
        targetText: string,
        currentChannelId?: string | null
    ): Promise<ResolvedChannelTarget> {
        const query = targetText.trim();
        const entries = await this.listGuildStructure(guild);
        const exactIdMatchEntries = entries.filter((entry) => entry.id === query);
        if (exactIdMatchEntries.length) {
            const resolvedIds = this.expandToMessageChannelIds(exactIdMatchEntries, entries);
            return {
                query,
                resolvedIds,
                entries: exactIdMatchEntries,
                exactIdMatch: true,
                confidence: "exact",
            };
        }

        const normalizedTarget = normalizeLookupValue(query);
        const scored = entries
            .map((entry) => {
                const normalizedName = normalizeLookupValue(entry.name);
                const normalizedParent = normalizeLookupValue(entry.parentCategoryName || "");
                let score = 0;

                if (normalizedName === normalizedTarget) {
                    score += 8;
                }
                if (normalizedName.includes(normalizedTarget)) {
                    score += 5;
                }
                if (normalizedTarget.includes(normalizedName) && normalizedName) {
                    score += 3;
                }
                if (normalizedParent && normalizedParent.includes(normalizedTarget)) {
                    score += 2;
                }
                if (currentChannelId && entry.id === currentChannelId) {
                    score += 1;
                }

                return { entry, score };
            })
            .filter((item) => item.score > 0)
            .sort((left, right) => right.score - left.score || left.entry.name.localeCompare(right.entry.name));

        const selectedEntries = scored.slice(0, 3).map((item) => item.entry);
        return {
            query,
            resolvedIds: this.expandToMessageChannelIds(selectedEntries, entries),
            entries: selectedEntries,
            exactIdMatch: false,
            confidence:
                scored[0]?.score >= 8 ? "exact" : scored[0]?.score >= 5 ? "high" : scored[0]?.score >= 3 ? "medium" : "low",
        };
    }

    public static async getGuildCompletenessSummary(guild: Guild | null): Promise<{
        liveReadableChannelCount: number;
        liveReadableCategoryCount: number;
        cachedOnlyCount: number;
        indexedChannelCount: number;
        liveHydrated: boolean;
    }> {
        const entries = await this.listGuildStructure(guild);
        return {
            liveReadableChannelCount: entries.filter(
                (entry) => entry.source === "live" && entry.isReadable && isMessageChannelEntry(entry)
            ).length,
            liveReadableCategoryCount: entries.filter(
                (entry) => entry.source === "live" && entry.isViewable && isCategoryEntry(entry)
            ).length,
            cachedOnlyCount: entries.filter((entry) => entry.source === "cached_only").length,
            indexedChannelCount: entries.filter((entry) => entry.isIndexed).length,
            liveHydrated: Boolean(guild),
        };
    }

    private static expandToMessageChannelIds(
        selectedEntries: GuildStructureEntry[],
        allEntries: GuildStructureEntry[]
    ): string[] {
        const ids = new Set<string>();
        for (const entry of selectedEntries) {
            if (isCategoryEntry(entry)) {
                for (const child of allEntries) {
                    if (child.parentCategoryId === entry.id && isMessageChannelEntry(child)) {
                        ids.add(child.id);
                    }
                }
                continue;
            }
            ids.add(entry.id);
        }
        return [...ids];
    }
}

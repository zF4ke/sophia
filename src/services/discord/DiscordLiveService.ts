import type {
    Channel,
    Client,
    Guild,
    GuildMember,
    GuildTextBasedChannel,
} from "discord.js";

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
    ): Promise<{
        id: string;
        username: string;
        displayName: string;
        roles: string[];
    } | null> {
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

        return {
            id: member.id,
            username: member.user.username,
            displayName: member.displayName,
            roles: member.roles.cache
                .filter((role) => role.name !== "@everyone")
                .map((role) => role.name)
                .slice(0, 10),
        };
    }

    public static async listMembers(
        guild: Guild | null,
        filters?: string
    ): Promise<Array<{ id: string; username: string; displayName: string }>> {
        if (!guild) {
            return [];
        }

        await guild.members.fetch();
        const normalized = filters?.trim().toLowerCase();

        return guild.members.cache
            .filter((member) => {
                if (!normalized) {
                    return true;
                }

                return (
                    member.user.username.toLowerCase().includes(normalized) ||
                    member.displayName.toLowerCase().includes(normalized)
                );
            })
            .first(12)
            .map((member) => ({
                id: member.id,
                username: member.user.username,
                displayName: member.displayName,
            }));
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

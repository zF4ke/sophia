import { PermissionFlagsBits, type Guild, type Client, type GuildMember } from "discord.js";
import { AccessPolicy } from "./AccessPolicy";

export interface SourceAudience { client?: Client; privateResponse?: boolean; destinationChannelId?: string | null }

export async function assertReadableChannels(guild: Guild | null, actorId: string | null | undefined, channelIds: string[], audience: SourceAudience = {}): Promise<void> {
    if (!channelIds.length) return;
    if (!actorId) throw new Error("Source history requires an authenticated requester.");
    const members = new Map<string, GuildMember>();
    for (const id of new Set(channelIds)) {
        let channel = guild ? await guild.channels.fetch(id, { force: true }).catch(() => null) : null;
        if (!channel && audience.privateResponse) {
            const fetched = await (audience.client ?? guild?.client)?.channels.fetch(id, { force: true }).catch(() => null);
            if (fetched && "guild" in fetched) channel = fetched;
        }
        const sourceGuild = channel?.guild ?? guild;
        if (!sourceGuild) throw new Error("Source guild is unavailable.");
        if (sourceGuild.id !== guild?.id && await AccessPolicy.decide(actorId, sourceGuild, "none") === "deny") throw new Error("Source guild access is no longer granted.");
        if (!members.has(sourceGuild.id)) members.set(sourceGuild.id, await sourceGuild.members.fetch({ user: actorId, force: true }));
        const member = members.get(sourceGuild.id)!;
        if (!channel || !channel.isTextBased() || !channel.permissionsFor(member)?.has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.ReadMessageHistory])) throw new Error(`History is not available to this requester in channel ${id}.`);
        if (!audience.privateResponse && audience.destinationChannelId && audience.destinationChannelId !== id &&
            !channel.permissionsFor(sourceGuild.id)?.has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.ReadMessageHistory])) {
            throw new Error("This source has a restricted audience. Use a private response or continue in its source channel.");
        }
    }
}

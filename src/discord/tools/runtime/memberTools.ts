import type { Guild } from "discord.js";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import type { DiscordToolResult, MemberListSort } from "@/shared/appTypes";

export async function getMemberProfile(
    guild: Guild | null,
    nameOrId: string
): Promise<DiscordToolResult> {
    const profile = await DiscordLiveService.getMemberProfile(guild, nameOrId);
    return {
        tool: "get_member_profile",
        summary: profile
            ? `${profile.displayName} (@${profile.username}) com ${profile.roles.length} cargos visíveis.`
            : "Membro não encontrado.",
        data: profile,
    };
}

export async function listMembers(
    guild: Guild | null,
    options: {
        filters?: string;
        limit?: number;
        offset?: number;
        sort?: MemberListSort;
    } = {}
): Promise<DiscordToolResult> {
    const members = await DiscordLiveService.listMembers(guild, options);
    return {
        tool: "list_members",
        summary: members.returnedCount
            ? members.hasMore
                ? `Mostrando ${members.returnedCount} de ${members.totalCount} membros em ordem de entrada.`
                : `${members.returnedCount} membros listados em ordem de entrada.`
            : "Nenhum membro correspondente encontrado.",
        data: members,
    };
}

export async function getGuildContext(guild: Guild | null): Promise<DiscordToolResult> {
    const context = await DiscordLiveService.getGuildContext(guild);
    return {
        tool: "get_guild_context",
        summary: context
            ? `${context.name}: ${context.memberCount} membros e ${context.channelCount} canais.`
            : "Contexto do servidor indisponível.",
        data: context,
    };
}

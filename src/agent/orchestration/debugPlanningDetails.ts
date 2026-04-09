import type { Guild } from "discord.js";
import type { SearchPlan } from "@/shared/appTypes";
import { INTERACTIVE_CRAWL_LIMIT } from "@/discord/live/DiscordChannelCrawlService";

const SEARCH_MESSAGE_LIMIT = 30;

export function compactDebugText(value: string, maxLength: number): string {
    const compact = value.replace(/\s+/g, " ").trim();
    if (compact.length <= maxLength) {
        return compact;
    }

    return `${compact.slice(0, maxLength - 3)}...`;
}

export function describePlannedTool(
    plan: SearchPlan,
    guild: Guild | null,
    currentChannelId?: string | null
): string[] {
    const details: string[] = [];

    if (plan.action === "search_messages") {
        const query = String(plan.arguments.query || "").trim();
        const limit = Number(plan.arguments.limit || SEARCH_MESSAGE_LIMIT);
        const requestedScope = String(plan.arguments.scope || "guild");
        const location = guild
            ? requestedScope === "channel" && currentChannelId
                ? `memória local do servidor, com foco em <#${currentChannelId}>`
                : "memória local deste servidor"
            : "memória local disponível";

        details.push(`Onde: ${location}`);
        if (query) {
            details.push(`Busca: "${compactDebugText(query, 80)}"`);
        }
        details.push(`Limite: até ${limit} resultados`);
        return details;
    }

    if (plan.action === "list_relevant_channels") {
        const query = String(plan.arguments.query || "").trim();
        if (query) {
            details.push(`Tema: "${compactDebugText(query, 80)}"`);
        }
        details.push(guild ? "Origem: canais do servidor atual" : "Origem: canais disponíveis");
        return details;
    }

    if (plan.action === "crawl_channel_messages") {
        const channelId = String(plan.arguments.channelId || "").trim();
        const limit = Number(plan.arguments.limit || INTERACTIVE_CRAWL_LIMIT);
        if (channelId) {
            details.push(`Canal: <#${channelId}>`);
        }
        details.push(`Busca ao vivo: até ${limit} mensagens`);
        return details;
    }

    if (plan.action === "read_message_thread") {
        const messageId = String(plan.arguments.messageId || "").trim();
        if (messageId) {
            details.push(`Mensagem base: ${messageId}`);
        }
        return details;
    }

    if (plan.action === "read_channel_summary") {
        const channelId = String(plan.arguments.channelId || "").trim();
        if (channelId) {
            details.push(`Canal: <#${channelId}>`);
        }
        return details;
    }

    if (plan.action === "get_member_profile") {
        const nameOrId = String(plan.arguments.nameOrId || "").trim();
        if (nameOrId) {
            details.push(`Alvo: ${compactDebugText(nameOrId, 80)}`);
        }
        return details;
    }

    if (plan.action === "list_members") {
        const filters = String(plan.arguments.filters || "").trim();
        const limit = Number(plan.arguments.limit || 100);
        const offset = Number(plan.arguments.offset || 0);
        details.push("Origem: membros do servidor atual");
        if (filters) {
            details.push(`Filtro: ${compactDebugText(filters, 80)}`);
        }
        details.push(`Faixa: ${offset + 1} até ${offset + limit}`);
        details.push("Ordem: entrada no servidor");
        return details;
    }

    if (plan.action === "get_guild_context") {
        details.push("Origem: metadados do servidor atual");
        return details;
    }

    return details;
}

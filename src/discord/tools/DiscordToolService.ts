import type { Guild } from "discord.js";
import type {
    DiscordToolResult,
    MemberListSort,
} from "@/shared/appTypes";
import * as messageTools from "@/discord/tools/runtime/messageTools";
import * as channelDiscoveryTools from "@/discord/tools/runtime/channelDiscoveryTools";
import * as memberTools from "@/discord/tools/runtime/memberTools";
import * as resultReaders from "@/discord/tools/runtime/resultReaders";
import { withCachedToolResult } from "@/discord/tools/runtime/toolCache";

export class DiscordToolService {
    public static async searchMessages(
        question: string,
        guild: Guild | null,
        options: {
            limit?: number;
            channelIds?: string[];
            authorId?: string;
        } = {}
    ): Promise<DiscordToolResult> {
        return withCachedToolResult(
            "search_messages",
            guild?.id || null,
            {
                question,
                limit: options.limit ?? 8,
                channelIds: options.channelIds ?? [],
                authorId: options.authorId ?? null,
            },
            () => messageTools.searchMessages(question, guild, options)
        );
    }

    public static async readMessageThread(messageId: string): Promise<DiscordToolResult> {
        return withCachedToolResult(
            "read_message_thread",
            null,
            { messageId },
            () => messageTools.readMessageThread(messageId)
        );
    }

    public static async readChannelSummary(channelId: string): Promise<DiscordToolResult> {
        return withCachedToolResult(
            "read_channel_summary",
            null,
            { channelId },
            () => messageTools.readChannelSummary(channelId)
        );
    }

    public static async listRelevantChannels(
        query: string,
        guild: Guild | null,
        currentChannelId?: string | null
    ): Promise<DiscordToolResult> {
        return withCachedToolResult(
            "list_relevant_channels",
            guild?.id || null,
            {
                query,
                currentChannelId: currentChannelId ?? null,
            },
            () => channelDiscoveryTools.listRelevantChannels(query, guild, currentChannelId)
        );
    }

    public static async crawlChannelMessages(
        guild: Guild | null,
        channelId: string,
        limit = 1000,
        queryHint?: string
    ): Promise<DiscordToolResult> {
        return channelDiscoveryTools.crawlChannelMessages(guild, channelId, limit, queryHint);
    }

    public static async getMemberProfile(
        guild: Guild | null,
        nameOrId: string
    ): Promise<DiscordToolResult> {
        return withCachedToolResult(
            "get_member_profile",
            guild?.id || null,
            { nameOrId },
            () => memberTools.getMemberProfile(guild, nameOrId)
        );
    }

    public static async listMembers(
        guild: Guild | null,
        options: {
            filters?: string;
            limit?: number;
            offset?: number;
            sort?: MemberListSort;
        } = {}
    ): Promise<DiscordToolResult> {
        return withCachedToolResult(
            "list_members",
            guild?.id || null,
            {
                filters: options.filters ?? null,
                limit: options.limit ?? null,
                offset: options.offset ?? null,
                sort: options.sort ?? null,
            },
            () => memberTools.listMembers(guild, options)
        );
    }

    public static async getGuildContext(guild: Guild | null): Promise<DiscordToolResult> {
        return withCachedToolResult(
            "get_guild_context",
            guild?.id || null,
            { guildId: guild?.id || null },
            () => memberTools.getGuildContext(guild)
        );
    }

    public static extractBestChunk(results: DiscordToolResult[]) {
        return resultReaders.extractBestChunk(results);
    }

    public static getListMembersResult(result: DiscordToolResult) {
        return resultReaders.getListMembersResult(result);
    }

    public static getMemberProfileResult(result: DiscordToolResult) {
        return resultReaders.getMemberProfileResult(result);
    }

    public static getCrawlResult(result: DiscordToolResult) {
        return resultReaders.getCrawlResult(result);
    }
}

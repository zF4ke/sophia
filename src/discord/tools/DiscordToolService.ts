import type { Guild } from "discord.js";
import type {
    DiscordToolResult,
    MemberListSort,
} from "@/shared/appTypes";
import * as messageTools from "@/discord/tools/runtime/messageTools";
import * as channelDiscoveryTools from "@/discord/tools/runtime/channelDiscoveryTools";
import * as memberTools from "@/discord/tools/runtime/memberTools";
import * as resultReaders from "@/discord/tools/runtime/resultReaders";

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
        return messageTools.searchMessages(question, guild, options);
    }

    public static async readMessageThread(messageId: string): Promise<DiscordToolResult> {
        return messageTools.readMessageThread(messageId);
    }

    public static async readChannelSummary(channelId: string): Promise<DiscordToolResult> {
        return messageTools.readChannelSummary(channelId);
    }

    public static async listRelevantChannels(
        query: string,
        guild: Guild | null,
        currentChannelId?: string | null
    ): Promise<DiscordToolResult> {
        return channelDiscoveryTools.listRelevantChannels(query, guild, currentChannelId);
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
        return memberTools.getMemberProfile(guild, nameOrId);
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
        return memberTools.listMembers(guild, options);
    }

    public static async getGuildContext(guild: Guild | null): Promise<DiscordToolResult> {
        return memberTools.getGuildContext(guild);
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

import type { Guild } from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import type { DiscordToolResult, RetrievedChunk } from "@/shared/appTypes";

export class DiscordToolService {
    public static async searchMessages(
        question: string,
        guild: Guild | null,
        limit = 8
    ): Promise<DiscordToolResult> {
        const results = await DiscordMemoryService.searchMessagesAsync(
            question,
            { guildId: guild?.id || null },
            limit
        );

        return {
            tool: "search_messages",
            summary: results.length
                ? `Found ${results.length} relevant message chunks.`
                : "No relevant stored messages found.",
            data: results,
        };
    }

    public static async readMessageThread(messageId: string): Promise<DiscordToolResult> {
        const thread = DiscordMemoryService.getMessageThread(messageId);
        return {
            tool: "read_message_thread",
            summary: thread.length
                ? `Loaded ${thread.length} nearby thread messages.`
                : "No nearby thread context found.",
            data: thread,
        };
    }

    public static async readChannelSummary(channelId: string): Promise<DiscordToolResult> {
        const summary = DiscordMemoryService.getChannelSummary(channelId);
        return {
            tool: "read_channel_summary",
            summary: summary
                ? `Loaded summary for channel ${channelId}.`
                : "No stored summary for that channel.",
            data: summary,
        };
    }

    public static async listRelevantChannels(
        query: string,
        guild: Guild | null
    ): Promise<DiscordToolResult> {
        const channels = DiscordMemoryService.listRelevantChannels(query, {
            guildId: guild?.id || null,
        });

        return {
            tool: "list_relevant_channels",
            summary: channels.length
                ? `Found ${channels.length} potentially relevant channels.`
                : "No relevant channels found in local memory.",
            data: channels,
        };
    }

    public static async getMemberProfile(
        guild: Guild | null,
        nameOrId: string
    ): Promise<DiscordToolResult> {
        const profile = await DiscordLiveService.getMemberProfile(guild, nameOrId);
        return {
            tool: "get_member_profile",
            summary: profile ? `Loaded member profile for ${profile.displayName}.` : "Member not found.",
            data: profile,
        };
    }

    public static async listMembers(
        guild: Guild | null,
        filters?: string
    ): Promise<DiscordToolResult> {
        const members = await DiscordLiveService.listMembers(guild, filters);
        return {
            tool: "list_members",
            summary: members.length ? `Loaded ${members.length} members.` : "No matching members found.",
            data: members,
        };
    }

    public static async getGuildContext(guild: Guild | null): Promise<DiscordToolResult> {
        const context = await DiscordLiveService.getGuildContext(guild);
        return {
            tool: "get_guild_context",
            summary: context ? `Loaded guild context for ${context.name}.` : "No guild context available.",
            data: context,
        };
    }

    public static extractBestChunk(results: DiscordToolResult[]): RetrievedChunk | null {
        for (const result of results) {
            if (result.tool !== "search_messages") {
                continue;
            }

            const chunks = result.data as RetrievedChunk[];
            if (chunks.length) {
                return chunks[0];
            }
        }

        return null;
    }
}

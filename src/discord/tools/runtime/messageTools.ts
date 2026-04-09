import type { Guild } from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { DiscordToolResult } from "@/shared/appTypes";

export async function searchMessages(
    question: string,
    guild: Guild | null,
    options: {
        limit?: number;
        channelIds?: string[];
        authorId?: string;
    } = {}
): Promise<DiscordToolResult> {
    const results = await DiscordMemoryService.searchMessagesAsync(
        question,
        {
            guildId: guild?.id || null,
            channelIds: options.channelIds,
            authorIds: options.authorId ? [options.authorId] : undefined,
        },
        options.limit ?? 8
    );

    return {
        tool: "search_messages",
        summary: results.length
            ? `Found ${results.length} relevant message chunks.`
            : "No relevant stored messages found.",
        data: results,
    };
}

export async function readMessageThread(messageId: string): Promise<DiscordToolResult> {
    const thread = DiscordMemoryService.getMessageThread(messageId);
    return {
        tool: "read_message_thread",
        summary: thread.length
            ? `Loaded ${thread.length} nearby thread messages.`
            : "No nearby thread context found.",
        data: thread,
    };
}

export async function readChannelSummary(channelId: string): Promise<DiscordToolResult> {
    const summary = DiscordMemoryService.getChannelSummary(channelId);
    return {
        tool: "read_channel_summary",
        summary: summary
            ? `Loaded summary for channel ${channelId}.`
            : "No stored summary for that channel.",
        data: summary,
    };
}

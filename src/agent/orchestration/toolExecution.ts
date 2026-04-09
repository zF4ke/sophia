import type { Guild } from "discord.js";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";
import { INTERACTIVE_CRAWL_LIMIT } from "@/discord/live/DiscordChannelCrawlService";
import type {
    DiscordToolResult,
    MemberListSort,
    SearchPlan,
} from "@/shared/appTypes";

const SEARCH_MESSAGE_LIMIT = 30;

export async function executeTool(
    plan: SearchPlan,
    guild: Guild | null,
    question: string,
    currentChannelId?: string | null,
    onProgress?: (toolName: string, summary: string) => Promise<void> | void
): Promise<DiscordToolResult> {
    try {
        switch (plan.action) {
            case "search_messages":
                return DiscordToolService.searchMessages(String(plan.arguments.query || question), guild, {
                    limit: Number(plan.arguments.limit || SEARCH_MESSAGE_LIMIT),
                    channelIds:
                        typeof plan.arguments.channelIds === "string"
                            ? String(plan.arguments.channelIds)
                                  .split(",")
                                  .map((value) => value.trim())
                                  .filter(Boolean)
                            : undefined,
                    authorId: plan.arguments.authorId
                        ? String(plan.arguments.authorId)
                        : undefined,
                });
            case "read_message_thread":
                return DiscordToolService.readMessageThread(String(plan.arguments.messageId || ""));
            case "read_channel_summary":
                return DiscordToolService.readChannelSummary(String(plan.arguments.channelId || ""));
            case "list_relevant_channels":
                return DiscordToolService.listRelevantChannels(
                    String(plan.arguments.query || question),
                    guild,
                    currentChannelId
                );
            case "crawl_channel_messages":
                return DiscordToolService.crawlChannelMessages(
                    guild,
                    String(plan.arguments.channelId || ""),
                    Number(plan.arguments.limit || INTERACTIVE_CRAWL_LIMIT),
                    plan.arguments.queryHint ? String(plan.arguments.queryHint) : undefined,
                    onProgress
                );
            case "get_member_profile":
                return DiscordToolService.getMemberProfile(
                    guild,
                    String(plan.arguments.nameOrId || "")
                );
            case "list_members":
                return DiscordToolService.listMembers(guild, {
                    filters: plan.arguments.filters ? String(plan.arguments.filters) : undefined,
                    limit: plan.arguments.limit ? Number(plan.arguments.limit) : undefined,
                    offset: plan.arguments.offset ? Number(plan.arguments.offset) : undefined,
                    sort: plan.arguments.sort
                        ? (String(plan.arguments.sort) as MemberListSort)
                        : undefined,
                });
            case "get_guild_context":
                return DiscordToolService.getGuildContext(guild);
            case "finish":
            default:
                return {
                    tool: "finish",
                    summary: "Stopped search loop.",
                    data: null,
                };
        }
    } catch (error) {
        const errorMessage =
            error instanceof Error && error.message ? error.message : "Tool execution failed.";
        return {
            tool: plan.action,
            summary: `Falhou: ${errorMessage}`,
            data: null,
            errorMessage,
        };
    }
}

import type {
    AnswerCitation,
    ChannelCrawlResult,
    DiscordToolResult,
    LiveMemberListResult,
    MemberProfileResult,
    RetrievedChunk,
} from "@/shared/appTypes";
import type {
    ChannelSummaryEvidence,
    StoredMessageEvidence,
} from "@/agent/orchestration/types";

export function getToolEvidenceCount(result: DiscordToolResult): number {
    switch (result.tool) {
        case "search_messages":
        case "read_message_thread":
            return Array.isArray(result.data) ? result.data.length : 0;
        case "list_members": {
            const members = result.data as LiveMemberListResult | null;
            return members?.returnedCount ?? 0;
        }
        case "read_channel_summary": {
            const summary = result.data as ChannelSummaryEvidence | null;
            return summary ? summary.recentMessages.length : 0;
        }
        case "get_member_profile":
        case "get_guild_context":
            return result.data ? 1 : 0;
        case "crawl_channel_messages": {
            const crawl = result.data as ChannelCrawlResult;
            return crawl?.previewMessages?.length ?? 0;
        }
        default:
            return 0;
    }
}

export function getDebugItemCount(result: DiscordToolResult): number | undefined {
    if (Array.isArray(result.data)) {
        return result.data.length;
    }

    if (result.tool === "list_members") {
        return (result.data as LiveMemberListResult | null)?.returnedCount;
    }

    if (result.tool === "crawl_channel_messages") {
        return (result.data as ChannelCrawlResult)?.previewMessages?.length;
    }

    return undefined;
}

export function buildCitations(toolRuns: DiscordToolResult[]): AnswerCitation[] {
    const citations: AnswerCitation[] = [];
    const seenLinks = new Set<string>();

    const pushCitation = (label: string, jumpLink?: string) => {
        if (!jumpLink || seenLinks.has(jumpLink) || citations.length >= 3) {
            return;
        }

        seenLinks.add(jumpLink);
        citations.push({ label, jumpLink });
    };

    for (const run of toolRuns) {
        if (run.tool === "search_messages") {
            (run.data as RetrievedChunk[]).forEach((item) => {
                pushCitation(`${item.channelName} · ${item.authorName}`, item.jumpLink);
            });
            continue;
        }

        if (run.tool === "read_message_thread") {
            (run.data as StoredMessageEvidence[]).forEach((item) => {
                pushCitation(`${item.channelName} · ${item.authorName}`, item.jumpLink);
            });
            continue;
        }

        if (run.tool === "read_channel_summary") {
            const summary = run.data as ChannelSummaryEvidence | null;
            summary?.recentMessages.forEach((item) => {
                pushCitation(`${item.channelName} · ${item.authorName}`, item.jumpLink);
            });
        }
    }

    return citations;
}

export function formatEvidence(toolRuns: DiscordToolResult[]): string {
    const lines: string[] = [];

    for (const run of toolRuns) {
        lines.push(`Tool: ${run.tool}`);
        lines.push(`Summary: ${run.summary}`);

        if (run.tool === "search_messages") {
            const chunks = run.data as RetrievedChunk[];
            chunks.slice(0, 5).forEach((chunk, index) => {
                lines.push(
                    `${index + 1}. [${chunk.channelName}] ${chunk.authorName}: ${chunk.content} (${chunk.jumpLink})`
                );
            });
        } else if (run.tool === "read_message_thread") {
            const thread = run.data as StoredMessageEvidence[];
            thread.slice(0, 6).forEach((message, index) => {
                lines.push(
                    `${index + 1}. [${message.channelName}] ${message.authorName}: ${message.content} (${message.jumpLink})`
                );
            });
        } else if (run.tool === "read_channel_summary") {
            const summary = run.data as ChannelSummaryEvidence | null;
            if (summary) {
                lines.push(`Channel: ${summary.channelId}`);
                lines.push(`Stored messages: ${summary.messageCount}`);
                summary.recentMessages.slice(0, 5).forEach((message, index) => {
                    lines.push(
                        `${index + 1}. [${message.channelName}] ${message.authorName}: ${message.content} (${message.jumpLink})`
                    );
                });
            }
        } else if (run.tool === "get_guild_context") {
            const context = run.data as
                | {
                      id: string;
                      name: string;
                      memberCount: number;
                      channelCount: number;
                  }
                | null;
            if (context) {
                lines.push(`Server name: ${context.name}`);
                lines.push(`Members: ${context.memberCount}`);
                lines.push(`Channels: ${context.channelCount}`);
            }
        } else if (run.tool === "get_member_profile") {
            const profile = run.data as MemberProfileResult | null;
            if (profile) {
                lines.push(`Display name: ${profile.displayName}`);
                lines.push(`Username: @${profile.username}`);
                if (profile.globalName) {
                    lines.push(`Global name: ${profile.globalName}`);
                }
                if (profile.nickname) {
                    lines.push(`Nickname: ${profile.nickname}`);
                }
                lines.push(
                    `Roles: ${profile.roles.length ? profile.roles.join(", ") : "none visible"}`
                );
                if (profile.bio) {
                    lines.push(`Bio: ${profile.bio}`);
                }
            }
        } else if (run.tool === "list_members") {
            const members = run.data as LiveMemberListResult;
            if (!members) {
                lines.push("Member list unavailable.");
                lines.push("");
                continue;
            }
            lines.push(`Total members available: ${members.totalCount}`);
            lines.push(
                `Members shown: ${members.returnedCount} (offset ${members.offset}, limit ${members.limit})`
            );
            lines.push(`Has more: ${members.hasMore ? "yes" : "no"}`);
            members.members.slice(0, 50).forEach((member, index) => {
                const position = members.offset + index + 1;
                lines.push(`${position}. ${member.displayName} (@${member.username})`);
            });
        } else if (run.tool === "crawl_channel_messages") {
            const crawl = run.data as ChannelCrawlResult;
            lines.push(`Channel: ${crawl.channelName}`);
            lines.push(`Fetched: ${crawl.messagesFetched}`);
            lines.push(`Stored: ${crawl.messagesStored}`);
            lines.push(`Exhausted: ${crawl.exhausted ? "yes" : "no"}`);
            if (crawl.backgroundIngestQueued) {
                lines.push("Background indexing: queued");
            }
            crawl.previewMessages?.slice(0, 6).forEach((message, index) => {
                lines.push(
                    `${index + 1}. [${crawl.channelName}] ${message.authorName}: ${message.content} (${message.jumpLink})`
                );
            });
        } else {
            lines.push(String(run.data ?? "No data."));
        }

        lines.push("");
    }

    return lines.join("\n").trim();
}

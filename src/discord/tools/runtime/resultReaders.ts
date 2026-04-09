import type {
    ChannelCrawlResult,
    DiscordToolResult,
    LiveMemberListResult,
    MemberProfileResult,
    RetrievedChunk,
} from "@/shared/appTypes";

export function extractBestChunk(results: DiscordToolResult[]): RetrievedChunk | null {
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

export function getListMembersResult(result: DiscordToolResult): LiveMemberListResult | null {
    if (result.tool !== "list_members") {
        return null;
    }

    return result.data as LiveMemberListResult;
}

export function getMemberProfileResult(result: DiscordToolResult): MemberProfileResult | null {
    if (result.tool !== "get_member_profile") {
        return null;
    }

    return result.data as MemberProfileResult | null;
}

export function getCrawlResult(result: DiscordToolResult): ChannelCrawlResult | null {
    if (result.tool !== "crawl_channel_messages") {
        return null;
    }

    return result.data as ChannelCrawlResult;
}

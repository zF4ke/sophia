import type {
    ChannelCrawlResult,
    DiscordToolResult,
    LiveMemberListResult,
    RetrievedChunk,
} from "@/shared/appTypes";
import {
    DISCORD_TOOL_EVIDENCE_ROLES,
    type DiscordToolName,
} from "@/shared/discordTools";
import {
    buildCitations,
    formatEvidence,
    getToolEvidenceCount,
} from "@/agent/orchestration/evidenceFormatting";
import {
    extractRequestedOrdinal,
    isMemberDiscoveryQuestion,
    isPersonMessageQuestion,
    requestsAllMembers,
} from "@/agent/orchestration/questionAnalysis";
import type {
    AggregatedMemberEvidence,
    GroundingAssessment,
} from "@/agent/orchestration/types";

const MIN_SEARCH_SCORE = 0.15;

export function assessGrounding(
    question: string,
    toolRuns: DiscordToolResult[]
): GroundingAssessment {
    const usefulRuns = toolRuns.filter((run) => {
        if (run.tool === "finish") {
            return false;
        }

        if (run.tool === "crawl_channel_messages") {
            const crawl = run.data as ChannelCrawlResult;
            return Boolean(crawl?.previewMessages?.length);
        }

        const role = DISCORD_TOOL_EVIDENCE_ROLES[run.tool as DiscordToolName];
        if (role === "discovery_only") {
            return false;
        }

        if (run.tool === "search_messages") {
            const chunks = run.data as RetrievedChunk[];
            return Boolean(chunks.length && chunks[0].totalScore >= MIN_SEARCH_SCORE);
        }

        return getToolEvidenceCount(run) > 0;
    });

    const messageEvidenceCount = usefulRuns
        .filter(
            (run) =>
                run.tool === "crawl_channel_messages" ||
                DISCORD_TOOL_EVIDENCE_ROLES[run.tool as DiscordToolName] ===
                    "message_evidence"
        )
        .reduce((total, run) => total + getToolEvidenceCount(run), 0);
    const liveEvidenceCount = usefulRuns
        .filter(
            (run) =>
                DISCORD_TOOL_EVIDENCE_ROLES[run.tool as DiscordToolName] === "live_evidence"
        )
        .reduce((total, run) => total + getToolEvidenceCount(run), 0);
    const sufficient = isGroundingSufficient(question, usefulRuns);

    return {
        summary: {
            messageEvidenceCount,
            liveEvidenceCount,
            sufficient,
        },
        evidence: formatEvidence(usefulRuns),
        citations: buildCitations(usefulRuns),
    };
}

export function isGroundingSufficient(
    question: string,
    toolRuns: DiscordToolResult[]
): boolean {
    if (!toolRuns.length) {
        return false;
    }

    const messageEvidenceCount = toolRuns
        .filter(
            (run) =>
                DISCORD_TOOL_EVIDENCE_ROLES[run.tool as DiscordToolName] ===
                "message_evidence"
        )
        .reduce((total, run) => total + getToolEvidenceCount(run), 0);
    const liveEvidenceCount = toolRuns
        .filter(
            (run) =>
                DISCORD_TOOL_EVIDENCE_ROLES[run.tool as DiscordToolName] === "live_evidence"
        )
        .reduce((total, run) => total + getToolEvidenceCount(run), 0);

    if (isMemberDiscoveryQuestion(question)) {
        const memberEvidence = collectMemberEvidence(toolRuns);
        if (!memberEvidence) {
            return false;
        }

        return !memberEvidenceNeedsMore(
            question,
            memberEvidence,
            extractRequestedOrdinal(question)
        );
    }

    if (isPersonMessageQuestion(question)) {
        return messageEvidenceCount > 0;
    }

    return messageEvidenceCount > 0 || liveEvidenceCount > 0;
}

export function collectMemberEvidence(
    toolRuns: DiscordToolResult[]
): AggregatedMemberEvidence | null {
    const listRuns = toolRuns
        .filter((run) => run.tool === "list_members")
        .map((run) => run.data as LiveMemberListResult | null)
        .filter((run): run is LiveMemberListResult => Boolean(run));

    if (!listRuns.length) {
        return null;
    }

    const byId = new Map<string, LiveMemberListResult["members"][number]>();
    let totalCount = 0;
    let sort: LiveMemberListResult["sort"] = "joined_at";
    let filters: string | null = null;

    listRuns.forEach((run) => {
        totalCount = Math.max(totalCount, run.totalCount);
        sort = run.sort;
        filters = run.filters;
        run.members.forEach((member) => {
            if (!byId.has(member.id)) {
                byId.set(member.id, member);
            }
        });
    });

    const members = [...byId.values()].sort((left, right) => {
        const leftJoined = left.joinedTimestamp ?? Number.MAX_SAFE_INTEGER;
        const rightJoined = right.joinedTimestamp ?? Number.MAX_SAFE_INTEGER;
        if (leftJoined !== rightJoined) {
            return leftJoined - rightJoined;
        }
        return left.id.localeCompare(right.id);
    });

    return {
        members,
        totalCount,
        hasMore: members.length < totalCount,
        sort,
        filters,
    };
}

export function memberEvidenceNeedsMore(
    question: string,
    evidence: AggregatedMemberEvidence,
    ordinal: number | null
): boolean {
    if (ordinal !== null) {
        return evidence.members.length < ordinal && evidence.hasMore;
    }

    if (requestsAllMembers(question)) {
        return evidence.hasMore;
    }

    return false;
}

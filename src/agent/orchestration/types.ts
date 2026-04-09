import type {
    AnswerCitation,
    DiscordToolResult,
    GroundingDecisionMode,
    GroundingSummary,
    LiveMemberListResult,
} from "@/shared/appTypes";
import type { ReusableGroundedContextRecord } from "@/memory/types";

export type StoredMessageEvidence = {
    channelId: string;
    channelName: string;
    authorName: string;
    content: string;
    jumpLink: string;
};

export type ChannelSummaryEvidence = {
    channelId: string;
    messageCount: number;
    recentMessages: StoredMessageEvidence[];
};

export type AggregatedMemberEvidence = {
    members: LiveMemberListResult["members"];
    totalCount: number;
    hasMore: boolean;
    sort: LiveMemberListResult["sort"];
    filters: string | null;
};

export type GroundingAssessment = {
    summary: GroundingSummary;
    evidence: string;
    citations: AnswerCitation[];
};

export type GroundingDecision = {
    sufficient: boolean;
    mode: GroundingDecisionMode;
    reason: string;
    missingInformation: string | null;
};

export type ReusableGroundedContext = ReusableGroundedContextRecord;

export type SearchContext = {
    crawledChannelIds: Set<string>;
    seededFromContext: boolean;
    initialToolRuns: DiscordToolResult[];
};

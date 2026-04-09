import type {
    AnswerCitation,
    ConversationResolutionContext,
    DiscordToolResult,
    GroundingDecisionMode,
    RouteIntent,
} from "@/shared/appTypes";

export interface StoredMessage {
    id: string;
    guildId: string | null;
    channelId: string;
    channelName: string;
    authorId: string;
    authorName: string;
    content: string;
    attachmentsJson: string;
    referenceMessageId: string | null;
    createdTimestamp: number;
    jumpLink: string;
    isBot: number;
}

export interface ChannelIndexState {
    channelId: string;
    lastMessageId: string | null;
    lastIndexedTimestamp: number | null;
}

export interface SearchMessageScope {
    guildId?: string | null;
    channelIds?: string[];
    authorIds?: string[];
}

export interface ChannelIndexState {
    channelId: string;
    lastMessageId: string | null;
    lastIndexedTimestamp: number | null;
}

export interface ChannelCrawlState {
    channelId: string;
    lastCrawledTimestamp: number | null;
    oldestFetchedMessageId: string | null;
    exhausted: boolean;
}

export interface CachedToolResultRecord {
    cacheKey: string;
    toolName: string;
    guildId: string | null;
    argumentsJson: string;
    result: DiscordToolResult;
    createdTimestamp: number;
    expiryTimestamp: number;
    createdResponseOrdinal: number | null;
}

export interface ReusableGroundedContextRecord {
    guildId: string | null;
    channelId: string | null;
    channelScopeKey: string;
    questionFingerprint: string;
    routeIntent: RouteIntent;
    evidenceText: string;
    citations: AnswerCitation[];
    toolRuns: DiscordToolResult[];
    sufficient: boolean;
    groundingDecisionMode: GroundingDecisionMode;
    createdTimestamp: number;
    expiryTimestamp: number;
    createdResponseOrdinal: number | null;
}

export interface ConversationResolutionContextRecord extends ConversationResolutionContext {}

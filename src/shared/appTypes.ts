import type {
    AutocompleteInteraction,
    ButtonInteraction,
    ChatInputCommandInteraction,
    Client,
    Collection,
    ModalSubmitInteraction,
    StringSelectMenuInteraction,
} from "discord.js";
import type { DiscordToolName } from "@/shared/discordTools";

export type SupportedInteraction =
    | ChatInputCommandInteraction
    | AutocompleteInteraction
    | ButtonInteraction
    | StringSelectMenuInteraction
    | ModalSubmitInteraction;

export interface ModelProfile {
    chatModel: string;
    analysisModel: string;
    embeddingModel: string;
    temperature: number;
    maxOutputTokens: number;
    webEnabled?: boolean;
}

export type WebMode = "off" | "auto" | "required";
export type WebStatus = "off" | "enabled" | "used";

export interface ModelProfileConfig {
    defaultProfile: string;
    profiles: Record<string, ModelProfile>;
}

export interface AppConfig {
    discordToken: string;
    openRouterApiKey: string;
    openRouterBaseUrl: string;
    modelProfileName: string;
    modelProfile: ModelProfile;
    runtime: {
        operationalDbPath: string;
        checkpointDbPath: string;
        maxToolCalls: number;
        maxResearchPasses: number;
        maxRepeatedCallSignature: number;
        maxLatencyBudgetMs: number;
        maxPriorTurns: number;
        maxChannelMessages: number;
        maxToolRunsContext: number;
        maxEvidenceSlice: number;
        interactiveCrawlLimit: number;
        escalationFetchLimit: number;
        retrievalHistoryLimit: number;
        retrievalContextWindow: number;
    };
}

export interface BotCommand {
    data: {
        name: string;
        toJSON(): unknown;
    };
    execute(interaction: ChatInputCommandInteraction, client: BotClient): Promise<void>;
    autocomplete?(
        interaction: AutocompleteInteraction,
        client: BotClient
    ): Promise<void>;
}

export type BotClient = Client & {
    commands: Collection<string, BotCommand>;
};

export interface RetrievedChunk {
    messageId: string;
    channelId: string;
    channelName: string;
    guildId: string | null;
    authorId: string;
    authorName: string;
    authorUsername?: string | null;
    authorNickname?: string | null;
    content: string;
    createdTimestamp: number;
    jumpLink: string;
    lexicalScore: number;
    semanticScore: number;
    recencyScore: number;
    totalScore: number;
}

export type RetrievalMode = "history" | "semantic" | "mixed";

export interface SemanticContinuationCursor {
    lastScore: number;
    lastCreatedTimestamp: number;
    lastMessageId: string;
}

export interface RetrievalContinuation {
    history: {
        perChannelOldestMessageId: Record<string, string | null>;
        continuationAvailable: boolean;
    };
    semantic: {
        cursor: SemanticContinuationCursor | null;
        continuationAvailable: boolean;
    };
    perChannelOldestMessageId: Record<string, string | null>;
    continuationAvailable: boolean;
}

export interface RetrievalExhaustion {
    historyExhaustedChannelIds: string[];
    historyExhausted: boolean;
    semanticExhausted: boolean;
    exhaustedChannelIds: string[];
    exhausted: boolean;
}

export interface RetrievalAccumulatedWindow {
    beforeTimestamp: number | null;
    afterTimestamp: number | null;
}

export interface MultiLaneRetrievalResult {
    query: string;
    mode: RetrievalMode;
    historyMessages: RetrievedChunk[];
    semanticMatches: RetrievedChunk[];
    combinedResults: RetrievedChunk[];
    cacheHit: boolean;
    liveEscalated: boolean;
    searchedChannelIds: string[];
    fetchedChannelIds: string[];
    cacheEnriched: boolean;
    evidenceSufficient: boolean;
    strongResultCount: number;
    weakResultCount: number;
    historyMessageCount: number;
    semanticMatchCount: number;
    sourceOrigin: "none" | "cache" | "live_refresh" | "cache_after_refresh";
    targetAuthorId: string | null;
    targetChannelIds: string[];
    continuation: RetrievalContinuation;
    exhaustion: RetrievalExhaustion;
    accumulatedWindow: RetrievalAccumulatedWindow;
    accumulatedUniqueCount: number;
    beforeTimestamp: number | null;
    afterTimestamp: number | null;
    excludedMessageIds: string[];
    retrievalDiagnostics?: {
        strictScopedQuery: boolean;
        continuationInputsApplied: boolean;
        scopedEmptyRetryAttempted: boolean;
        scopedEmptyRetryRecovered: boolean;
        retryStrategy: "none" | "without_excluded" | "without_cursor";
    };
}

export interface ChannelCandidate {
    channelId: string;
    channelName: string;
    hitCount: number;
    isIndexed: boolean;
    matchSource: "memory" | "live_name";
    lastIndexedTimestamp: number | null;
}

export interface AnswerCitation {
    label: string;
    jumpLink: string;
}

export type GroundedAnswerMode = "confident" | "best_effort" | "insufficient";
export type MemberListSort = "joined_at";

export interface ModelTraceContext {
    traceLabel: string;
    questionPreview?: string;
    webMode?: WebMode;
    webContext?: string;
}

export interface LiveMemberRecord {
    id: string;
    username: string;
    displayName: string;
    joinedTimestamp: number | null;
    globalName?: string | null;
    nickname?: string | null;
    isBot?: boolean;
}

export interface LiveMemberListResult {
    members: LiveMemberRecord[];
    totalCount: number;
    returnedCount: number;
    hasMore: boolean;
    offset: number;
    limit: number;
    sort: MemberListSort;
    filters: string | null;
}

export interface MemberProfileResult {
    id: string;
    query?: string;
    username: string;
    displayName: string;
    globalName: string | null;
    nickname: string | null;
    roles: string[];
    joinedAt: string | null;
    joinedTimestamp: number | null;
    accountCreatedAt: string | null;
    avatarUrl: string | null;
    premiumSince: string | null;
    pending: boolean;
    bannerUrl: string | null;
    accentColor: string | null;
    bio: string | null;
    isBot?: boolean;
    isCurrentGuildMember?: boolean;
    source?: "live_id" | "live_exact" | "live_search" | "historical_author";
    confidence?: "exact" | "high" | "medium";
}

export interface ResolvedMemberIdentity {
    query: string;
    resolvedId: string;
    displayName: string;
    username: string;
    globalName: string | null;
    nickname: string | null;
    isBot: boolean;
    isCurrentGuildMember: boolean;
    source: "live_id" | "live_exact" | "live_search" | "historical_author";
    confidence: "exact" | "high" | "medium";
    roles: string[];
}

export interface GuildStructureEntry {
    id: string;
    guildId: string | null;
    name: string;
    type: string;
    parentCategoryId: string | null;
    parentCategoryName: string | null;
    isReadable: boolean;
    isViewable: boolean;
    isIndexed: boolean;
    source: "live" | "cached_only";
    missingOrDeletedPossible: boolean;
}

export interface ResolvedChannelTarget {
    query: string;
    resolvedIds: string[];
    entries: GuildStructureEntry[];
    exactIdMatch: boolean;
    confidence: "exact" | "high" | "medium" | "low";
}

export interface ChannelCrawlResult {
    channelId: string;
    channelName: string;
    messagesFetched: number;
    messagesStored: number;
    exhausted: boolean;
    oldestFetchedMessageId?: string | null;
    queryHint: string | null;
    backgroundIngestQueued?: boolean;
    previewMessages?: Array<{
        messageId: string;
        authorId: string;
        authorName: string;
        authorUsername?: string | null;
        authorNickname?: string | null;
        content: string;
        createdTimestamp: number;
        jumpLink: string;
    }>;
}

export interface DiscordToolResult {
    tool: DiscordToolName | "finish";
    summary: string;
    data: unknown;
    cacheStatus?: "hit" | "miss";
    errorMessage?: string | null;
}

export interface RequestClassification {
    mode: "direct_answer" | "discord_grounded";
    reason: string;
}

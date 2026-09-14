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
    api?: "chat-completions" | "responses";
    reasoningEffort?: "minimal" | "low" | "medium" | "high";
    inputModalities?: Array<"text" | "image" | "video" | "audio">;
    label?: string;
    chatModel: string;
    embeddingModel: string;
    temperature: number;
    maxOutputTokens: number;
    contextWindow: number;
    parallelToolCalls?: boolean;
    /**
     * Optional OpenAI-compatible endpoint override for this profile (e.g. a
     * local LM Studio server at http://127.0.0.1:1234/v1). When set, chat
     * requests for this profile go there instead of OpenRouter, and the
     * OpenRouter-only `provider` routing block is omitted.
     */
    baseUrl?: string;
    /**
     * Optional env var name holding the API key for a `baseUrl` profile
     * (e.g. "OPENCODE_API_KEY" for OpenCode Zen). When unset, "not-needed"
     * is sent as the key (local servers don't check it).
     */
    apiKeyEnv?: string;
    /** Optional provider hint (e.g. "opencode" for Zen free models). */
    provider?: string;
    /** Free-form operator notes shown nowhere at runtime. */
    notes?: string;
    pricing?: {
        inputPerMillionUsd?: number;
        outputPerMillionUsd?: number;
        cacheReadPerMillionUsd?: number;
        webSearchPerCallUsd?: number;
        source?: string;
        updatedAt?: string;
    };
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
        toolCallLimit: number;
        maxRepeatedCallSignature: number;
        maxPriorTurns: number;
        maxChannelMessages: number;
        maxToolRunsContext: number;
        maxEvidenceSlice: number;
        escalationFetchLimit: number;
        retrievalHistoryLimit: number;
        retrievalContextWindow: number;
        longTask: {
            evidenceSliceFloor: number;
        };
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
    traceEvents?: Array<{ label: string; detail: string; timestamp: number }>;
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
    /** When the member started boosting this guild (NOT Nitro status). Null if not boosting. */
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
    channelTopic?: string | null;
    type: string;
    position: number | null;
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
    /** Visual inputs are passed to the next model turn; durable records keep their source labels. */
    images?: Array<{ url: string; label: string }>;
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

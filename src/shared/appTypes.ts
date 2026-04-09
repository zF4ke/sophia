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

export interface ModelProfileConfig {
    defaultProfile: string;
    profiles: Record<string, ModelProfile>;
}

export interface AppConfig {
    discordToken: string;
    openRouterApiKey: string;
    openRouterBaseUrl: string;
    port: number;
    modelProfileName: string;
    modelProfile: ModelProfile;
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
    content: string;
    createdTimestamp: number;
    jumpLink: string;
    lexicalScore: number;
    semanticScore: number;
    recencyScore: number;
    totalScore: number;
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

export interface GroundingSummary {
    messageEvidenceCount: number;
    liveEvidenceCount: number;
    sufficient: boolean;
}

export type GroundingDecisionMode = "heuristic" | "judge" | "reused";

export interface EvidenceJudgeResult {
    sufficient: boolean;
    reason: string;
    missingInformation: string | null;
}

export type RouteIntent =
    | "channel_target"
    | "person_target"
    | "member_lookup"
    | "server_context"
    | "broad_search";

export interface RouteDecision {
    source: "deterministic" | "ai";
    intent: RouteIntent;
    targetText: string | null;
    channelIds?: string[];
    authorId?: string;
    authorQuery?: string;
    topicText?: string | null;
    channelHintText?: string | null;
    resolvedPerson?: ResolvedPersonContext | null;
    confidence: number;
    reason: string;
}

export interface ModelTraceContext {
    traceLabel: string;
    questionPreview?: string;
}

export type MemberListSort = "joined_at";

export interface LiveMemberRecord {
    id: string;
    username: string;
    displayName: string;
    joinedTimestamp: number | null;
    globalName?: string | null;
    nickname?: string | null;
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
    username: string;
    displayName: string;
    globalName: string | null;
    nickname: string | null;
    roles: string[];
    bannerUrl: string | null;
    accentColor: string | null;
    bio: string | null;
}

export interface ResolvedPersonContext {
    id: string;
    username: string;
    displayName: string;
    globalName: string | null;
    nickname: string | null;
    roles: string[];
}

export interface ConversationResolutionContext {
    guildId: string | null;
    channelId: string | null;
    routeIntent: RouteIntent;
    targetText: string | null;
    authorId: string | null;
    authorQuery: string | null;
    channelIds: string[];
    topicText: string | null;
    channelHintText: string | null;
    resolvedPerson: ResolvedPersonContext | null;
    createdTimestamp: number;
    expiryTimestamp: number;
    createdResponseOrdinal: number | null;
}

export interface ChannelCrawlResult {
    channelId: string;
    channelName: string;
    messagesFetched: number;
    messagesStored: number;
    exhausted: boolean;
    queryHint: string | null;
}

export interface DiscordToolResult {
    tool: DiscordToolName | "finish";
    summary: string;
    data: unknown;
    cacheStatus?: "hit" | "miss";
}

export interface SearchPlan {
    action: DiscordToolName | "finish";
    arguments: Record<string, string | number | undefined>;
    reason: string;
}

export interface RequestClassification {
    mode: "direct_answer" | "discord_grounded";
    reason: string;
}

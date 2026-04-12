import type { Guild, Message, User } from "discord.js";
import type { z } from "zod";
import type {
    DiscordToolResult,
    GuildStructureEntry,
    GroundedAnswerMode,
    RetrievalMode,
    SemanticContinuationCursor,
    RequestClassification,
    ResolvedChannelTarget,
    ResolvedMemberIdentity,
    WebMode,
    WebStatus,
} from "@/shared/appTypes";
import type { DiscordToolEvidenceRole, DiscordToolName } from "@/shared/discordTools";

export type TurnTrigger = "talk" | "mention" | "reply";
export type RuntimeMode = "conversation" | "research" | "refusal";
export type StopReason =
    | "direct_answer"
    | "evidence_sufficient"
    | "budget_exhausted"
    | "confidence_plateau"
    | "no_useful_next_step"
    | "insufficient_evidence";
export type RetrievalSourceOrigin = "none" | "cache" | "live_refresh" | "cache_after_refresh";
export type EvidenceStrength = "strong" | "weak" | "metadata";
export type ToolArgumentValue =
    | string
    | number
    | boolean
    | null
    | ToolArgumentValue[]
    | { [key: string]: ToolArgumentValue };
export type ToolArguments = Record<string, ToolArgumentValue | undefined>;

export interface ReplyContext {
    messageId: string;
    authorId: string;
    authorName: string;
    authorDisplayName: string;
    content: string;
    jumpLink?: string | null;
}

export interface ConversationContext {
    key: string;
    kind: "native_thread" | "reply_chain" | "channel";
    trigger: TurnTrigger;
    replyAnchorMessageId?: string | null;
    nativeThreadId?: string | null;
}

export interface ChannelContextMessage {
    authorName: string;
    content: string;
    createdTimestamp: number;
}

export interface ConversationTurnSummary {
    requestId: string;
    question: string;
    answer: string;
    classificationMode: string;
    runtimeMode: string;
    stopReason: string;
    confidence: string;
    createdTimestamp: number;
}

export interface TurnInput {
    question: string;
    user: User;
    requesterDisplayName: string;
    guild: Guild | null;
    currentChannelId?: string | null;
    nativeThreadId?: string | null;
    debugSession?: RuntimeDebugSession | null;
    requestedWebMode?: WebMode;
    trigger: TurnTrigger;
    replyContext?: ReplyContext | null;
    referencedMessage?: Message | null;
    conversation: ConversationContext;
}

export interface CapabilityManifest {
    id: DiscordToolName;
    kind: "tool";
    description: string;
    inputSchema: z.ZodTypeAny;
    outputSchema: z.ZodTypeAny;
    sideEffectLevel: "none";
    authRequirements: string[];
    costClass: "cheap" | "normal" | "expensive";
    latencyClass: "fast" | "medium" | "slow";
    evidenceRole: DiscordToolEvidenceRole;
    preconditions: string[];
    postconditions: string[];
}

export interface RetrievalSummary {
    mode: RetrievalMode;
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
    accumulatedUniqueCount: number;
    sourceOrigin: RetrievalSourceOrigin;
    continuationAvailable: boolean;
    historyContinuationAvailable: boolean;
    semanticContinuationAvailable: boolean;
    historyCursorByChannel: Record<string, string | null>;
    semanticCursor: SemanticContinuationCursor | null;
    exhaustedChannelIds: string[];
    historyExhausted: boolean;
    semanticExhausted: boolean;
    beforeTimestamp: number | null;
    afterTimestamp: number | null;
    activeChannelIds: string[];
}

export interface ToolInvocationRecord {
    tool: DiscordToolName;
    arguments: ToolArguments;
    summary: string;
    learned: string;
    confidenceImproved: boolean;
    output: DiscordToolResult;
    durationMs: number;
    retrievalSummary?: RetrievalSummary | null;
    blocked?: boolean;
}

export interface EvidenceItem {
    tool: DiscordToolName;
    summary: string;
    content: string;
    evidenceRole: DiscordToolEvidenceRole;
    strength: EvidenceStrength;
    sourceOrigin: RetrievalSourceOrigin;
    messageId?: string | null;
    authorId?: string | null;
    authorName?: string | null;
    authorUsername?: string | null;
    authorNickname?: string | null;
    channelId?: string | null;
    channelName?: string | null;
    jumpLink?: string | null;
    createdTimestamp?: number | null;
}

export interface ActiveRetrievalSession {
    mode: RetrievalMode;
    channelIds: string[];
    authorId: string | null;
    beforeTimestamp: number | null;
    afterTimestamp: number | null;
    historyCursorByChannel: Record<string, string | null>;
    semanticCursor: SemanticContinuationCursor | null;
    seenMessageIds: string[];
    accumulatedUniqueCount: number;
    exhaustedChannelIds: string[];
    historyExhausted: boolean;
    semanticExhausted: boolean;
    continuationAvailable: boolean;
}

export interface RuntimeTraceEvent {
    label: string;
    detail: string;
    timestamp: number;
}

export interface GraphState {
    requestId: string;
    threadId: string;
    guildId: string | null;
    channelId: string | null;
    actorId: string;
    requesterDisplayName: string;
    trigger: TurnTrigger;
    conversationKind: ConversationContext["kind"];
    question: string;
    replyContext: ReplyContext | null;
    recentTurns: ConversationTurnSummary[];
    channelContext: ChannelContextMessage[];
    permissionContext: {
        isAdmin: boolean;
        canReadChannel: boolean;
        canReadHistory: boolean;
        canSendMessages: boolean;
    };
    requestedWebMode: WebMode;
    mode: RuntimeMode | null;
    classification: RequestClassification | null;
    goal: string;
    successCriteria: string;
    candidateCapabilities: DiscordToolName[];
    activeMemberTarget: ResolvedMemberIdentity | null;
    activeChannelTarget: ResolvedChannelTarget | null;
    activeResolvedChannelIds: string[];
    activeRetrievalSession: ActiveRetrievalSession | null;
    toolHistory: ToolInvocationRecord[];
    evidence: EvidenceItem[];
    retrievalSummary: RetrievalSummary | null;
    turnIntent: TurnIntent | null;
    stopReason: StopReason | null;
    confidence: GroundedAnswerMode;
    responseDraft: string | null;
    traceEvents: RuntimeTraceEvent[];
    constraints: {
        maxToolCalls: number;
        maxResearchPasses: number;
        maxRepeatedCallSignature: number;
        maxLatencyBudgetMs: number;
        maxPriorTurns: number;
        maxChannelMessages: number;
        maxToolRunsContext: number;
        maxEvidenceSlice: number;
    };
}

export interface GuildStructureFocusSummary {
    query: string | null;
    focusedEntries: GuildStructureEntry[];
    focusedResolvedIds: string[];
}

export interface TurnIntent {
    continuation: boolean;
    retrievalMode: RetrievalMode | null;
    beforeTimestamp: number | null;
    afterTimestamp: number | null;
    source: {
        continuation: "deterministic" | "model";
        retrievalMode: "deterministic" | "model" | "session" | "none";
        timeBounds: "deterministic" | "model" | "none";
    };
}

export interface PlanDecision {
    mode: RuntimeMode;
    reason: string;
    goal: string;
    successCriteria: string;
    candidateCapabilities: DiscordToolName[];
    confidence: GroundedAnswerMode;
    intent: TurnIntent;
}

export interface EvidenceDecision {
    sufficient: boolean;
    confidence: GroundedAnswerMode;
    reason: string;
}

export interface StepDecision {
    nextCapability: DiscordToolName | null;
    arguments: ToolArguments;
    reason: string;
    learnedExpectation: string;
}

export interface RuntimeAnswer {
    requestId: string;
    answer: string;
    threadId: string;
    citations: Array<{ label: string; jumpLink: string }>;
    classification: RequestClassification;
    toolRuns: DiscordToolResult[];
    confidence: GroundedAnswerMode;
}

export interface RuntimeDebugSession {
    setClassifying(): Promise<void>;
    setClassification(mode: "direct_answer" | "discord_grounded"): Promise<void>;
    setRequesterContext?(requesterLabel: string, trigger: TurnTrigger): Promise<void>;
    setConversationContext?(context: {
        threadId: string;
        kind: ConversationContext["kind"];
        replyAnchorMessageId?: string | null;
        replyContext?: ReplyContext | null;
    }): Promise<void>;
    setRuntimeMode?(mode: RuntimeMode): Promise<void>;
    setCheckpointThread?(threadId: string): Promise<void>;
    setPlanning(step: number): Promise<void>;
    setToolRunning(toolName: string, details?: string[]): Promise<void>;
    setToolProgress?(toolName: string, summary: string): Promise<void>;
    setToolResult(toolName: string, summary: string, itemCount?: number): Promise<void>;
    setRetrievalSummary?(summary: RetrievalSummary): Promise<void>;
    setGroundingSummary(
        summary: {
            messageEvidenceCount: number;
            liveEvidenceCount: number;
            sufficient: boolean;
        },
        decisionMode?: string,
        answerMode?: GroundedAnswerMode
    ): Promise<void>;
    setStopReason?(reason: StopReason, detail?: string | null): Promise<void>;
    setConfidence?(confidence: GroundedAnswerMode): Promise<void>;
    setWebStatus?(status: WebStatus): Promise<void>;
    setContextPreview?(preview: {
        recentChannelMessages: string[];
        evidencePreview: string[];
        recentTurns: string[];
    }): Promise<void>;
    setTraceEvent?(label: string, detail: string): Promise<void>;
    setGenerating(): Promise<void>;
    finishSuccess(summary?: string): Promise<void>;
    finishError(error: unknown): Promise<void>;
}

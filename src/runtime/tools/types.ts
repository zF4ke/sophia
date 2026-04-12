import type {
    ActiveRetrievalSession,
    EvidenceItem,
    RetrievalSummary,
    ToolArguments,
    ToolInvocationRecord,
    TurnIntent,
} from "@/runtime/contracts";
import type {
    DiscordToolResult,
    ResolvedChannelTarget,
    ResolvedMemberIdentity,
    RetrievalMode,
    SemanticContinuationCursor,
} from "@/shared/appTypes";
import type { DiscordToolName } from "@/shared/discordTools";

// ── Strategy Interface ──────────────────────────────────────────────

export interface ToolEnrichmentResult {
    arguments: ToolArguments;
    reason: string;
    learnedExpectation: string;
}

export interface EvidenceExtractionLimits {
    maxResolveChannelTargetEvidenceItems: number;
    maxRetrieveHistoryEvidenceItems: number;
    maxRetrieveSemanticEvidenceItems: number;
    maxRetrieveEvidenceContentChars: number;
}

export interface ToolStrategy {
    readonly id: DiscordToolName;

    /** Convert raw tool output into evidence items for the judge and synthesizer. */
    extractEvidence(run: DiscordToolResult, limits?: EvidenceExtractionLimits): EvidenceItem[];

    /** Enrich/normalize the model-provided arguments using resolved state. */
    enrichArguments(
        modelArgs: ToolArguments,
        modelStep: { reason: string; learnedExpectation: string },
        context: ArgumentEnrichmentContext
    ): ToolEnrichmentResult;

    /** Extract a resolved member identity from this tool's output, if applicable. */
    extractResolvedMember?(run: DiscordToolResult): ResolvedMemberIdentity | null;

    /** Extract a resolved channel target from this tool's output, if applicable. */
    extractResolvedChannel?(run: DiscordToolResult): ResolvedChannelTarget | null;

    /** Extract a retrieval summary from this tool's output. Only for retrieve_messages. */
    extractRetrievalSummary?(run: DiscordToolResult): RetrievalSummary | null;

    /** Extract an active retrieval session from this tool's output. Only for retrieve_messages. */
    extractRetrievalSession?(run: DiscordToolResult): ActiveRetrievalSession | null;
}

// ── Shared Context ──────────────────────────────────────────────────

export interface ArgumentEnrichmentContext {
    question: string;
    actorId: string;

    resolvedMember: { resolvedId: string | null; isCurrentGuildMember: boolean } | null;
    resolvedChannelIds: string[];
    structuralMember: string | null;
    structuralChannel: string | null;
    activeMember: ResolvedMemberIdentity | null;
    activeChannelTarget: { query: string; resolvedIds: string[] } | null;
    activeRetrievalSession: ActiveRetrievalSession | null;

    timeBounds: { beforeTimestamp?: number; afterTimestamp?: number };
    explicitBeforeTimestamp?: number;
    explicitAfterTimestamp?: number;

    shouldContinueSession: boolean;
    shouldAutoInjectCursor: boolean;

    turnIntent: TurnIntent | null;

    ambiguousMemberCandidate: { displayName: string; identifiers: string[] } | null;
    profiledMemberIdentifiers: Set<string> | null;

    channelMentionIds: string[];
}

// ── Payload Types ───────────────────────────────────────────────────

export type RetrievalPayload = {
    mode?: RetrievalMode;
    historyMessages?: Array<Record<string, unknown>>;
    semanticMatches?: Array<Record<string, unknown>>;
    combinedResults?: Array<Record<string, unknown>>;
    sourceOrigin?: RetrievalSummary["sourceOrigin"];
    targetAuthorId?: string | null;
    targetChannelIds?: string[];
    cacheHit?: boolean;
    liveEscalated?: boolean;
    searchedChannelIds?: string[];
    fetchedChannelIds?: string[];
    cacheEnriched?: boolean;
    evidenceSufficient?: boolean;
    strongResultCount?: number;
    weakResultCount?: number;
    historyMessageCount?: number;
    semanticMatchCount?: number;
    accumulatedUniqueCount?: number;
    continuation?: {
        history?: {
            perChannelOldestMessageId?: Record<string, string | null>;
            continuationAvailable?: boolean;
        };
        semantic?: {
            cursor?: SemanticContinuationCursor | null;
            continuationAvailable?: boolean;
        };
        perChannelOldestMessageId?: Record<string, string | null>;
        continuationAvailable?: boolean;
    };
    exhaustion?: {
        historyExhaustedChannelIds?: string[];
        historyExhausted?: boolean;
        semanticExhausted?: boolean;
        exhaustedChannelIds?: string[];
        exhausted?: boolean;
    };
    beforeTimestamp?: number | null;
    afterTimestamp?: number | null;
    excludedMessageIds?: string[];
};

export type GuildStructurePayload = {
    query?: string | null;
    entries?: Array<Record<string, unknown>>;
    focusedEntries?: Array<Record<string, unknown>>;
    focusedResolvedIds?: string[];
};

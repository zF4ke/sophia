import { randomUUID } from "crypto";
import { Annotation, END, START, StateGraph, task } from "@langchain/langgraph";
import { getAppConfig } from "@/app/AppConfig";
import { ModelGateway } from "@/ai/ModelGateway";
import { CapabilityRegistry } from "@/discord/capabilities/CapabilityRegistry";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import {
    answerConfidenceForInsufficient,
    classify,
    countEvidence,
    formatChannelContext,
    judgeEvidence,
    planNextStep,
    planWithModel,
    summarizeEvidence,
} from "@/runtime/planning";
import {
    buildConversationalRecovery,
    buildDirectConversationFallback,
    sanitizeConversationalAnswer,
} from "@/runtime/conversationRecovery";
import { PromptRegistry } from "@/runtime/PromptRegistry";
import { CheckpointStore } from "@/runtime/storage/CheckpointStore";
import { decideConversationWebMode } from "@/runtime/webFallback";
import type {
    ActiveRetrievalSession,
    EvidenceItem,
    GraphState,
    RetrievalSummary,
    RuntimeAnswer,
    RuntimeMode,
    RuntimeTraceEvent,
    StopReason,
    ToolArguments,
    ToolInvocationRecord,
    TurnInput,
} from "@/runtime/contracts";
import { DISCORD_TOOL_EVIDENCE_ROLES, type DiscordToolName } from "@/shared/discordTools";
import type {
    DiscordToolResult,
    GroundedAnswerMode,
    GuildStructureEntry,
    ResolvedChannelTarget,
    ResolvedMemberIdentity,
    RetrievalMode,
    SemanticContinuationCursor,
} from "@/shared/appTypes";

const State = Annotation.Root({
    requestId: Annotation<string>,
    threadId: Annotation<string>,
    guildId: Annotation<string | null>,
    channelId: Annotation<string | null>,
    actorId: Annotation<string>,
    requesterDisplayName: Annotation<string>,
    trigger: Annotation<TurnInput["trigger"]>,
    conversationKind: Annotation<TurnInput["conversation"]["kind"]>,
    question: Annotation<string>,
    replyContext: Annotation<GraphState["replyContext"]>,
    recentTurns: Annotation<GraphState["recentTurns"]>,
    channelContext: Annotation<GraphState["channelContext"]>,
    permissionContext: Annotation<GraphState["permissionContext"]>,
    requestedWebMode: Annotation<GraphState["requestedWebMode"]>,
    mode: Annotation<RuntimeMode | null>,
    classification: Annotation<GraphState["classification"]>,
    goal: Annotation<string>,
    successCriteria: Annotation<string>,
    candidateCapabilities: Annotation<GraphState["candidateCapabilities"]>,
    activeMemberTarget: Annotation<GraphState["activeMemberTarget"]>,
    activeChannelTarget: Annotation<GraphState["activeChannelTarget"]>,
    activeResolvedChannelIds: Annotation<GraphState["activeResolvedChannelIds"]>,
    activeRetrievalSession: Annotation<GraphState["activeRetrievalSession"]>,
    toolHistory: Annotation<ToolInvocationRecord[]>,
    evidence: Annotation<GraphState["evidence"]>,
    retrievalSummary: Annotation<GraphState["retrievalSummary"]>,
    turnIntent: Annotation<GraphState["turnIntent"]>,
    stopReason: Annotation<StopReason | null>,
    confidence: Annotation<GroundedAnswerMode>,
    responseDraft: Annotation<string | null>,
    traceEvents: Annotation<GraphState["traceEvents"]>,
    constraints: Annotation<GraphState["constraints"]>,
});

type RuntimeState = typeof State.State;

type RetrievalPayload = {
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

type GuildStructurePayload = {
    query?: string | null;
    entries?: Array<Record<string, unknown>>;
    focusedEntries?: Array<Record<string, unknown>>;
    focusedResolvedIds?: string[];
};

function normalizeLookupValue(value: string): string {
    return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, " ")
        .trim();
}

function isCategoryStructureEntry(entry: Pick<GuildStructureEntry, "type">): boolean {
    return entry.type === "4" || entry.type.toLowerCase().includes("category");
}

function asGuildStructureEntries(value: unknown): GuildStructureEntry[] {
    if (!Array.isArray(value)) {
        return [];
    }

    return value
        .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
        .map((item) => ({
            id: String(item.id || ""),
            guildId: item.guildId == null ? null : String(item.guildId),
            name: String(item.name || ""),
            type: String(item.type || "unknown"),
            parentCategoryId: item.parentCategoryId == null ? null : String(item.parentCategoryId),
            parentCategoryName:
                item.parentCategoryName == null ? null : String(item.parentCategoryName),
            isReadable: Boolean(item.isReadable),
            isViewable: Boolean(item.isViewable),
            isIndexed: Boolean(item.isIndexed),
            source: (item.source === "cached_only" ? "cached_only" : "live") as
                | "live"
                | "cached_only",
            missingOrDeletedPossible: Boolean(item.missingOrDeletedPossible),
        }))
        .filter((entry) => entry.id && entry.name);
}

function selectFocusedStructureEntries(
    entries: GuildStructureEntry[],
    query: string | null
): GuildStructureEntry[] {
    if (!query) {
        return [];
    }

    const normalizedQuery = normalizeLookupValue(query);
    if (!normalizedQuery) {
        return [];
    }

    return entries
        .map((entry) => {
            const normalizedName = normalizeLookupValue(entry.name);
            const normalizedParent = normalizeLookupValue(entry.parentCategoryName || "");
            let score = 0;

            if (normalizedName === normalizedQuery) {
                score += 10;
            }
            if (normalizedName && normalizedQuery.includes(normalizedName)) {
                score += 6;
            }
            if (normalizedName.includes(normalizedQuery)) {
                score += 5;
            }
            if (normalizedParent && normalizedQuery.includes(normalizedParent)) {
                score += 3;
            }
            if (normalizedParent && normalizedParent.includes(normalizedQuery)) {
                score += 2;
            }

            return { entry, score };
        })
        .filter((item) => item.score > 0)
        .sort((left, right) => right.score - left.score || left.entry.name.localeCompare(right.entry.name))
        .slice(0, 4)
        .map((item) => item.entry);
}

function buildStructureEvidenceItems(
    payload: GuildStructurePayload,
    summary: string
): EvidenceItem[] {
    const allEntries = asGuildStructureEntries(payload.entries);
    const focusedEntries = asGuildStructureEntries(payload.focusedEntries);
    const query = typeof payload.query === "string" && payload.query.trim() ? payload.query.trim() : null;
    const targets = focusedEntries.length ? focusedEntries : selectFocusedStructureEntries(allEntries, query);

    if (!targets.length) {
        return allEntries.slice(0, 8).map((entry) => ({
            tool: "list_guild_structure",
            summary,
            content: isCategoryStructureEntry(entry)
                ? `Category ${entry.name}. Viewable=${entry.isViewable ? "yes" : "no"}.`
                : `Channel #${entry.name}${entry.parentCategoryName ? ` in ${entry.parentCategoryName}` : ""}. Readable=${entry.isReadable ? "yes" : "no"}. Indexed=${entry.isIndexed ? "yes" : "no"}.`,
            evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
            strength: "metadata",
            sourceOrigin: "none",
            channelId: entry.id,
            channelName: entry.name,
        }));
    }

    const evidence: EvidenceItem[] = [];
    const seen = new Set<string>();

    for (const entry of targets) {
        if (seen.has(entry.id)) {
            continue;
        }
        seen.add(entry.id);

        if (isCategoryStructureEntry(entry)) {
            const children = allEntries.filter((candidate) => candidate.parentCategoryId === entry.id);
            const readableChildren = children.filter((candidate) => candidate.isReadable);
            const indexedChildren = readableChildren.filter((candidate) => candidate.isIndexed);
            const childLabels = readableChildren.slice(0, 6).map((candidate) => `#${candidate.name}`);
            const childPhrase = childLabels.length ? childLabels.join(", ") : "none";

            evidence.push({
                tool: "list_guild_structure",
                summary,
                content: `Category ${entry.name}. Visible channels under ${entry.name}: ${childPhrase}. Readable children: ${readableChildren.length}. Indexed children: ${indexedChildren.length}.`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
                strength: "metadata",
                sourceOrigin: "none",
                channelId: entry.id,
                channelName: entry.name,
            });

            for (const child of readableChildren.slice(0, 4)) {
                if (seen.has(child.id)) {
                    continue;
                }
                seen.add(child.id);
                evidence.push({
                    tool: "list_guild_structure",
                    summary,
                    content: `Channel #${child.name} in category ${entry.name}. Readable=yes. Indexed=${child.isIndexed ? "yes" : "no"}.`,
                    evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
                    strength: "metadata",
                    sourceOrigin: "none",
                    channelId: child.id,
                    channelName: child.name,
                });
            }
            continue;
        }

        evidence.push({
            tool: "list_guild_structure",
            summary,
            content: `Channel #${entry.name}${entry.parentCategoryName ? ` in category ${entry.parentCategoryName}` : ""}. Readable=${entry.isReadable ? "yes" : "no"}. Indexed=${entry.isIndexed ? "yes" : "no"}.`,
            evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
            strength: "metadata",
            sourceOrigin: "none",
            channelId: entry.id,
            channelName: entry.name,
        });
    }

    return evidence;
}

function extractResolvedMemberTarget(run: DiscordToolResult): ResolvedMemberIdentity | null {
    if (
        (run.tool !== "resolve_member_identity" && run.tool !== "get_member_profile") ||
        !run.data ||
        typeof run.data !== "object"
    ) {
        return null;
    }

    const item = run.data as Record<string, unknown>;
    const resolvedId =
        item.resolvedId == null ? (item.id == null ? null : String(item.id)) : String(item.resolvedId);
    if (!resolvedId) {
        return null;
    }

    return {
        query: String(item.query || resolvedId),
        resolvedId,
        displayName: String(item.displayName || "No Display Name"),
        username: String(item.username || "no username"),
        globalName: item.globalName == null ? null : String(item.globalName),
        nickname: item.nickname == null ? null : String(item.nickname),
        isBot: Boolean(item.isBot),
        isCurrentGuildMember:
            item.isCurrentGuildMember == null ? true : Boolean(item.isCurrentGuildMember),
        source:
            item.source === "historical_author" ||
            item.source === "live_exact" ||
            item.source === "live_search" ||
            item.source === "live_id"
                ? item.source
                : "live_id",
        confidence:
            item.confidence === "high" || item.confidence === "medium" || item.confidence === "exact"
                ? item.confidence
                : "medium",
        roles: Array.isArray(item.roles) ? item.roles.map(String) : [],
    };
}

function extractResolvedChannelTarget(run: DiscordToolResult): ResolvedChannelTarget | null {
    if (
        (run.tool !== "resolve_channel_targets" && run.tool !== "list_guild_structure") ||
        !run.data ||
        typeof run.data !== "object"
    ) {
        return null;
    }

    const item = run.data as GuildStructurePayload & Record<string, unknown>;
    const entries =
        run.tool === "resolve_channel_targets"
            ? asGuildStructureEntries(item.entries)
            : asGuildStructureEntries(item.focusedEntries);
    const resolvedIds =
        run.tool === "resolve_channel_targets"
            ? Array.isArray(item.resolvedIds)
                ? item.resolvedIds.map(String)
                : []
            : Array.isArray(item.focusedResolvedIds)
              ? item.focusedResolvedIds.map(String)
              : [];
    const query = typeof item.query === "string" && item.query.trim() ? item.query.trim() : "";

    if (!query && !entries.length && !resolvedIds.length) {
        return null;
    }

    return {
        query,
        resolvedIds,
        entries,
        exactIdMatch: Boolean(item.exactIdMatch),
        confidence:
            item.confidence === "exact" ||
            item.confidence === "high" ||
            item.confidence === "medium" ||
            item.confidence === "low"
                ? item.confidence
                : resolvedIds.length
                  ? "high"
                  : "low",
    };
}

function appendTrace(state: RuntimeState, label: string, detail: string): RuntimeTraceEvent[] {
    return state.traceEvents.concat({
        label,
        detail,
        timestamp: Date.now(),
    });
}

function argsSignature(tool: DiscordToolName, args: ToolArguments) {
    return `${tool}:${JSON.stringify(args)}`;
}

function getRetrievalSummary(run: DiscordToolResult): RetrievalSummary | null {
    if (run.tool !== "retrieve_messages" || !run.data || typeof run.data !== "object") {
        return null;
    }

    const payload = run.data as RetrievalPayload;
    return {
        mode: (payload.mode || "history") as RetrievalSummary["mode"],
        cacheHit: Boolean(payload.cacheHit),
        liveEscalated: Boolean(payload.liveEscalated),
        searchedChannelIds: (payload.searchedChannelIds || []).map(String),
        fetchedChannelIds: (payload.fetchedChannelIds || []).map(String),
        cacheEnriched: Boolean(payload.cacheEnriched),
        evidenceSufficient: Boolean(payload.evidenceSufficient),
        strongResultCount: Number(payload.strongResultCount || 0),
        weakResultCount: Number(payload.weakResultCount || 0),
        historyMessageCount: Number(payload.historyMessageCount || 0),
        semanticMatchCount: Number(payload.semanticMatchCount || 0),
        accumulatedUniqueCount: Number(payload.accumulatedUniqueCount || 0),
        sourceOrigin: (payload.sourceOrigin || "none") as RetrievalSummary["sourceOrigin"],
        continuationAvailable: Boolean(payload.continuation?.continuationAvailable),
        historyContinuationAvailable:
            payload.continuation?.history?.continuationAvailable == null
                ? Boolean(payload.continuation?.continuationAvailable)
                : Boolean(payload.continuation?.history?.continuationAvailable),
        historyCursorByChannel:
            payload.continuation?.history?.perChannelOldestMessageId ||
            payload.continuation?.perChannelOldestMessageId ||
            {},
        semanticContinuationAvailable: Boolean(payload.continuation?.semantic?.continuationAvailable),
        semanticCursor: payload.continuation?.semantic?.cursor || null,
        exhaustedChannelIds: (payload.exhaustion?.exhaustedChannelIds || []).map(String),
        historyExhausted: Boolean(payload.exhaustion?.historyExhausted),
        semanticExhausted: Boolean(payload.exhaustion?.semanticExhausted),
        beforeTimestamp:
            payload.beforeTimestamp == null ? null : Number(payload.beforeTimestamp),
        afterTimestamp:
            payload.afterTimestamp == null ? null : Number(payload.afterTimestamp),
        activeChannelIds: (payload.targetChannelIds || []).map(String),
    };
}

function extractActiveRetrievalSession(run: DiscordToolResult): ActiveRetrievalSession | null {
    if (run.tool !== "retrieve_messages" || !run.data || typeof run.data !== "object") {
        return null;
    }

    const payload = run.data as RetrievalPayload & Record<string, unknown>;
    const targetChannelIds = Array.isArray(payload.targetChannelIds)
        ? payload.targetChannelIds.map(String)
        : [];

    if (!targetChannelIds.length) {
        return null;
    }

    return {
        mode: (payload.mode || "history") as RetrievalMode,
        channelIds: targetChannelIds,
        authorId: payload.targetAuthorId == null ? null : String(payload.targetAuthorId),
        beforeTimestamp: payload.beforeTimestamp == null ? null : Number(payload.beforeTimestamp),
        afterTimestamp: payload.afterTimestamp == null ? null : Number(payload.afterTimestamp),
        historyCursorByChannel:
            payload.continuation?.history?.perChannelOldestMessageId ||
            payload.continuation?.perChannelOldestMessageId ||
            {},
        semanticCursor: payload.continuation?.semantic?.cursor || null,
        seenMessageIds: (payload.combinedResults || [])
            .map((row) => (row && typeof row === "object" && "messageId" in row ? String((row as Record<string, unknown>).messageId) : null))
            .filter((value): value is string => Boolean(value)),
        accumulatedUniqueCount: Number(payload.accumulatedUniqueCount || 0),
        exhaustedChannelIds: (payload.exhaustion?.exhaustedChannelIds || []).map(String),
        historyExhausted: Boolean(payload.exhaustion?.historyExhausted),
        semanticExhausted: Boolean(payload.exhaustion?.semanticExhausted),
        continuationAvailable: Boolean(payload.continuation?.continuationAvailable),
    };
}

function sameRetrievalScope(
    left: ActiveRetrievalSession | null,
    right: ActiveRetrievalSession | null
): boolean {
    if (!left || !right) {
        return false;
    }

    return (
        left.mode === right.mode &&
        left.authorId === right.authorId &&
        left.beforeTimestamp === right.beforeTimestamp &&
        left.afterTimestamp === right.afterTimestamp &&
        left.channelIds.length === right.channelIds.length &&
        left.channelIds.every((channelId, index) => channelId === right.channelIds[index])
    );
}

export function mergeActiveRetrievalSession(
    previous: ActiveRetrievalSession | null,
    next: ActiveRetrievalSession | null
): ActiveRetrievalSession | null {
    if (!next) {
        return previous;
    }
    if (!previous || !sameRetrievalScope(previous, next)) {
        return next;
    }

    const seenMessageIds = [...new Set([...previous.seenMessageIds, ...next.seenMessageIds])];
    return {
        ...next,
        historyCursorByChannel: {
            ...previous.historyCursorByChannel,
            ...next.historyCursorByChannel,
        },
        semanticCursor: next.semanticCursor || previous.semanticCursor,
        seenMessageIds,
        accumulatedUniqueCount: Math.max(
            previous.accumulatedUniqueCount,
            next.accumulatedUniqueCount,
            seenMessageIds.length
        ),
        exhaustedChannelIds: [...new Set([...previous.exhaustedChannelIds, ...next.exhaustedChannelIds])],
        historyExhausted: previous.historyExhausted || next.historyExhausted,
        semanticExhausted: previous.semanticExhausted || next.semanticExhausted,
        continuationAvailable:
            next.continuationAvailable ||
            previous.continuationAvailable,
    };
}

function extractEvidence(run: DiscordToolResult): EvidenceItem[] {
    if (run.tool === "retrieve_messages" && run.data && typeof run.data === "object") {
        const payload = run.data as RetrievalPayload;
        const historyRows = payload.historyMessages || [];
        const semanticRows = payload.semanticMatches || [];
        const sourceOrigin = (payload.sourceOrigin || "none") as RetrievalSummary["sourceOrigin"];
        const historyEvidence = historyRows.slice(0, 6).map((item) => {
            const body = String(item.content || "").slice(0, 260);
            const authorId = item.authorId == null ? null : String(item.authorId);
            const authorName = item.authorName == null ? null : String(item.authorName);
            const authorUsername = item.authorUsername == null ? null : String(item.authorUsername);
            const authorPrefix = authorName
                ? authorUsername && authorUsername !== authorName
                    ? `[${authorName} (@${authorUsername})${authorId ? ` id=${authorId}` : ""}]: `
                    : `[${authorName}${authorId ? ` id=${authorId}` : ""}]: `
                : authorId
                  ? `[id=${authorId}]: `
                  : "";
            const content = authorPrefix + body;
            return {
                tool: "retrieve_messages" as const,
                summary: run.summary,
                content,
                evidenceRole: "history_evidence" as const,
                strength: "strong" as const,
                sourceOrigin,
                authorId,
                authorName,
                authorUsername,
                channelId: item.channelId == null ? null : String(item.channelId),
                channelName: item.channelName == null ? null : String(item.channelName),
                jumpLink: item.jumpLink == null ? null : String(item.jumpLink),
                createdTimestamp: item.createdTimestamp == null ? null : Number(item.createdTimestamp),
            };
        });
        const semanticEvidence = semanticRows.slice(0, 6).map((item) => {
            const body = String(item.content || "").slice(0, 260);
            const lexicalScore = Number(item.lexicalScore || 0);
            const strength: EvidenceItem["strength"] =
                lexicalScore >= 2 || (lexicalScore >= 1 && body.length >= 80)
                    ? "strong"
                    : "weak";
            const authorId = item.authorId == null ? null : String(item.authorId);
            const authorName = item.authorName == null ? null : String(item.authorName);
            const authorUsername = item.authorUsername == null ? null : String(item.authorUsername);
            const authorPrefix = authorName
                ? authorUsername && authorUsername !== authorName
                    ? `[${authorName} (@${authorUsername})${authorId ? ` id=${authorId}` : ""}]: `
                    : `[${authorName}${authorId ? ` id=${authorId}` : ""}]: `
                : authorId
                  ? `[id=${authorId}]: `
                  : "";
            const content = authorPrefix + body;

            return {
                tool: "retrieve_messages" as const,
                summary: run.summary,
                content,
                evidenceRole: "semantic_evidence" as const,
                strength,
                sourceOrigin,
                authorId,
                authorName,
                authorUsername,
                channelId: item.channelId == null ? null : String(item.channelId),
                channelName: item.channelName == null ? null : String(item.channelName),
                jumpLink: item.jumpLink == null ? null : String(item.jumpLink),
                createdTimestamp: item.createdTimestamp == null ? null : Number(item.createdTimestamp),
            };
        });
        return [...historyEvidence, ...semanticEvidence];
    }

    if (run.tool === "resolve_member_identity" && run.data) {
        const item = run.data as Record<string, unknown>;
        const currentState =
            item.isCurrentGuildMember === false ? "historical guild memory" : "current guild";
        const resolvedId = item.resolvedId == null ? (item.id == null ? null : String(item.id)) : String(item.resolvedId);
        return [
            {
                tool: "resolve_member_identity",
                summary: run.summary,
                content: `${String(item.displayName || "No Display Name")} (@${String(item.username || "no username")}) from ${currentState}${resolvedId ? `; id=${resolvedId}` : ""}`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.resolve_member_identity,
                strength: "metadata",
                sourceOrigin: "none",
                authorId: resolvedId,
                authorName: item.displayName == null ? null : String(item.displayName),
            },
        ];
    }

    if (run.tool === "resolve_channel_targets" && run.data) {
        const item = run.data as { entries?: Array<Record<string, unknown>>; resolvedIds?: string[] };
        const entries = asGuildStructureEntries(item.entries);
        return entries.slice(0, 6).map((entry) => ({
            tool: "resolve_channel_targets",
            summary: run.summary,
            content: isCategoryStructureEntry(entry)
                ? `Category ${entry.name} resolved with ${Array.isArray(item.resolvedIds) ? item.resolvedIds.length : 0} visible message channels.`
                : `Channel #${entry.name}${entry.parentCategoryName ? ` in category ${entry.parentCategoryName}` : ""}.`,
            evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.resolve_channel_targets,
            strength: "metadata",
            sourceOrigin: "none",
            channelId: entry.id,
            channelName: entry.name,
        }));
    }

    if (run.tool === "list_guild_structure" && run.data) {
        return buildStructureEvidenceItems(run.data as GuildStructurePayload, run.summary);
    }

    if (run.tool === "get_member_profile" && run.data) {
        const item = run.data as Record<string, unknown>;
        const roles = Array.isArray(item.roles) ? item.roles.join(", ") : "none";
        const parts = [
            `${String(item.displayName || "No Display Name")} (@${String(item.username || "no username")})`,
        ];
        if (item.id != null) parts.push(`id=${String(item.id)}`);
        if (item.nickname) parts.push(`nick=${String(item.nickname)}`);
        if (item.joinedAt) parts.push(`joined=${String(item.joinedAt)}`);
        if (item.accountCreatedAt) parts.push(`created=${String(item.accountCreatedAt)}`);
        parts.push(`roles=${roles || "none"}`);
        if (item.isBot) parts.push("bot=true");
        if (item.premiumSince) parts.push(`nitro_since=${String(item.premiumSince)}`);
        if (item.pending) parts.push("pending=true");
        return [
            {
                tool: "get_member_profile",
                summary: run.summary,
                content: parts.join("; "),
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.get_member_profile,
                strength: "metadata",
                sourceOrigin: "none",
                authorId: item.id == null ? null : String(item.id),
                authorName: item.displayName == null ? null : String(item.displayName),
            },
        ];
    }

    if (run.tool === "list_members" && run.data && typeof run.data === "object") {
        const payload = run.data as { members?: Array<Record<string, unknown>>; hasMore?: boolean; totalCount?: number; offset?: number };
        const members = payload.members || [];
        const evidence = members.slice(0, 10).map((item) => {
            const parts = [
                `${String(item.displayName || "No Display Name")} (@${String(item.username || "no username")})`,
            ];
            if (item.id != null) parts.push(`id=${String(item.id)}`);
            if (item.nickname) parts.push(`nick=${String(item.nickname)}`);
            if (item.joinedTimestamp) {
                parts.push(`joined=${new Date(Number(item.joinedTimestamp)).toISOString()}`);
            }
            if (item.isBot) parts.push("bot=true");
            return {
                tool: "list_members" as const,
                summary: run.summary,
                content: parts.join("; "),
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_members,
                strength: "metadata" as const,
                sourceOrigin: "none" as const,
                authorId: item.id == null ? null : String(item.id),
                authorName: item.displayName == null ? null : String(item.displayName),
            };
        });
        if (payload.hasMore) {
            evidence.push({
                tool: "list_members" as const,
                summary: run.summary,
                content: `...and ${(payload.totalCount || 0) - members.length} more members (use offset=${(payload.offset || 0) + members.length} to continue).`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_members,
                strength: "metadata" as const,
                sourceOrigin: "none" as const,
                authorId: null,
                authorName: null,
            });
        }
        return evidence;
    }

    if (run.tool === "get_guild_context" && run.data) {
        const item = run.data as Record<string, unknown>;
        return [
            {
                tool: "get_guild_context",
                summary: run.summary,
                content: `${String(item.name || "Guild")}: ${String(item.memberCount || 0)} members, ${String(item.channelCount || 0)} channels`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.get_guild_context,
                strength: "metadata",
                sourceOrigin: "none",
            },
        ];
    }

    return [];
}

function summarizeRecentTurns(turns: GraphState["recentTurns"]): string {
    if (!turns.length) {
        return "No recent conversation turns.";
    }

    return turns
        .map(
            (turn, index) =>
                `${index + 1}. user=${turn.question} | sophia=${turn.answer} | mode=${turn.runtimeMode} | confidence=${turn.confidence} | stop=${turn.stopReason}`
        )
        .join("\n");
}

function buildFallbackAnswer(
    state: Pick<
        RuntimeState,
        "question" | "mode" | "confidence" | "evidence" | "replyContext" | "recentTurns" | "stopReason"
    >
): string {
    const fallbackConfidence =
        state.confidence === "insufficient" ? answerConfidenceForInsufficient(state) : state.confidence;

    if (
        state.mode === "conversation" &&
        !state.replyContext &&
        state.recentTurns.length === 0 &&
        state.evidence.length === 0
    ) {
        return buildDirectConversationFallback(state.question);
    }

    return buildConversationalRecovery({
        question: state.question,
        confidence: fallbackConfidence,
        evidence: state.evidence,
        replyContext: state.replyContext,
        priorTurns: state.recentTurns,
        stopReason: state.stopReason,
    });
}

const executeCapability = task(
    "execute_capability",
    async (
        tool: DiscordToolName,
        context: {
            guild: TurnInput["guild"];
            question: string;
            currentChannelId?: string | null;
            onProgress?: (toolName: string, summary: string) => Promise<void> | void;
        },
        args: ToolArguments
    ) => CapabilityRegistry.get(tool).run(context, args)
);

export class Runtime {
    private static graphPromise: Promise<any> | null = null;
    private static requestContext = new Map<string, TurnInput>();

    private static async getGraph() {
        if (!this.graphPromise) {
            this.graphPromise = this.buildGraph();
        }
        return this.graphPromise;
    }

    private static async buildGraph() {
        const checkpointer = CheckpointStore.getSaver();

        return new StateGraph(State)
            .addNode("ingest_turn", async (state: RuntimeState) => ({
                traceEvents: appendTrace(
                    state,
                    "ingest_turn",
                    `trigger=${state.trigger}; conversation=${state.conversationKind}; requester=${state.requesterDisplayName}`
                ),
            }))
            .addNode("load_context", async (state: RuntimeState) => ({
                traceEvents: appendTrace(
                    state,
                    "load_context",
                    `guild=${state.guildId || "dm"} channel=${state.channelId || "none"}`
                ),
            }))
            .addNode("load_checkpoint", async (state: RuntimeState) => ({
                traceEvents: appendTrace(state, "load_checkpoint", `thread=${state.threadId}`),
            }))
            .addNode("load_memory", async (state: RuntimeState) => {
                const recentTurns = await DiscordMemoryService.getRecentRuntimeRunsAsync(
                    state.threadId,
                    state.constraints.maxPriorTurns
                );
                const recentToolRuns = await DiscordMemoryService.getRecentToolRunsAsync(
                    state.threadId,
                    state.constraints.maxToolRunsContext
                );
                const channelMessages = state.channelId
                    ? await DiscordMemoryService.getRecentChannelMessagesAsync(state.channelId, state.constraints.maxChannelMessages)
                    : [];
                const channelContext = channelMessages.map((msg) => ({
                    authorName: msg.authorName,
                    content: msg.content.slice(0, 150),
                    createdTimestamp: msg.createdTimestamp,
                }));
                let activeMemberTarget = state.activeMemberTarget;
                let activeChannelTarget = state.activeChannelTarget;
                let activeResolvedChannelIds = [...state.activeResolvedChannelIds];
                let activeRetrievalSession = state.activeRetrievalSession;
                const reconstructedEvidence: EvidenceItem[] = [];
                const seenEvidence = new Set<string>();

                for (const run of [...recentToolRuns].reverse()) {
                    try {
                        const parsedOutput = JSON.parse(run.outputJson) as DiscordToolResult;
                        const resolvedMember = extractResolvedMemberTarget(parsedOutput);
                        const resolvedChannel = extractResolvedChannelTarget(parsedOutput);
                        const retrieval = getRetrievalSummary(parsedOutput);
                        const retrievalSession = extractActiveRetrievalSession(parsedOutput);
                        const evidenceItems = extractEvidence(parsedOutput);

                        if (resolvedMember) {
                            activeMemberTarget = resolvedMember;
                        }
                        if (resolvedChannel) {
                            activeChannelTarget = resolvedChannel;
                            activeResolvedChannelIds = resolvedChannel.resolvedIds;
                        } else if (retrieval?.searchedChannelIds?.length) {
                            activeResolvedChannelIds = retrieval.searchedChannelIds;
                        }
                        if (retrievalSession) {
                            activeRetrievalSession = mergeActiveRetrievalSession(
                                activeRetrievalSession,
                                retrievalSession
                            );
                        }

                        for (const item of evidenceItems) {
                            const key = [
                                item.tool,
                                item.jumpLink || "",
                                item.channelId || "",
                                item.authorId || "",
                                item.createdTimestamp || "",
                                item.content,
                            ].join("::");
                            if (seenEvidence.has(key)) {
                                continue;
                            }
                            seenEvidence.add(key);
                            reconstructedEvidence.push(item);
                        }
                    } catch {
                        continue;
                    }
                }
                const evidence = reconstructedEvidence.slice(-state.constraints.maxEvidenceSlice);
                const input = this.requestContext.get(state.requestId);
                await input?.debugSession?.setContextPreview?.({
                    recentChannelMessages: channelContext.map((m) => `${m.authorName}: ${m.content}`),
                    evidencePreview: evidence.slice(0, 6).map((item) => {
                        const channelLabel = item.channelName ? `#${item.channelName}` : "?";
                        const authorLabel = item.authorName
                            ? item.authorUsername && item.authorUsername !== item.authorName
                                ? `${item.authorName} (@${item.authorUsername})`
                                : item.authorName
                            : "?";
                        return `[${item.tool}] ${channelLabel} · ${authorLabel}: ${item.content}`;
                    }),
                    recentTurns: recentTurns.map((t) => `Q: ${t.question} | A: ${t.answer}`),
                });
                return {
                    recentTurns,
                    channelContext,
                    evidence,
                    activeMemberTarget,
                    activeChannelTarget,
                    activeResolvedChannelIds,
                    activeRetrievalSession,
                    traceEvents: appendTrace(
                        state,
                        "load_memory",
                        `Loaded ${recentTurns.length} prior conversation turn(s), ${channelContext.length} recent channel message(s), ${recentToolRuns.length} recent tool run(s), and reused ${evidence.length} evidence item(s).`
                    ),
                };
            })
            .addNode("plan_turn", async (state: RuntimeState) => {
                const input = this.requestContext.get(state.requestId);
                const plan = await planWithModel(
                    input || {
                        question: state.question,
                        user: { id: state.actorId } as TurnInput["user"],
                        requesterDisplayName: state.requesterDisplayName,
                        guild: null,
                        currentChannelId: state.channelId,
                        trigger: state.trigger,
                        requestedWebMode: state.requestedWebMode,
                        replyContext: state.replyContext,
                        conversation: {
                            key: state.threadId,
                            kind: state.conversationKind,
                            trigger: state.trigger,
                        },
                    },
                    state.channelContext,
                    state.recentTurns,
                    {
                        activeMemberTarget: state.activeMemberTarget,
                        activeChannelTarget: state.activeChannelTarget,
                        activeResolvedChannelIds: state.activeResolvedChannelIds,
                        activeRetrievalSession: state.activeRetrievalSession,
                    }
                );

                return {
                    mode: plan.mode,
                    classification: classify(plan.mode),
                    goal: plan.goal,
                    successCriteria: plan.successCriteria,
                    candidateCapabilities: plan.candidateCapabilities,
                    confidence: plan.confidence,
                    turnIntent: plan.intent,
                    traceEvents: appendTrace(state, "plan_turn", plan.reason),
                };
            })
            .addNode("route_mode", async (state: RuntimeState) => ({
                traceEvents: appendTrace(state, "route_mode", `mode=${state.mode || "conversation"}`),
            }))
            .addNode("run_research_loop", async (state: RuntimeState) => {
                const startedAt = Date.now();
                const toolHistory = [...state.toolHistory];
                const evidence = [...state.evidence];
                const traceEvents = [...state.traceEvents];
                let activeMemberTarget = state.activeMemberTarget;
                let activeChannelTarget = state.activeChannelTarget;
                let activeResolvedChannelIds = [...state.activeResolvedChannelIds];
                let activeRetrievalSession = state.activeRetrievalSession;
                let retrievalSummary = state.retrievalSummary;
                let confidence = state.confidence;
                let stopReason: StopReason = "insufficient_evidence";
                const repeated = new Map<string, number>();

                for (let pass = 0; pass < state.constraints.maxResearchPasses; pass += 1) {
                    const evidenceDecision = await judgeEvidence({
                        question: state.question,
                        toolHistory,
                        evidence,
                        actorId: state.actorId,
                        candidateCapabilities: state.candidateCapabilities,
                    });
                    traceEvents.push({
                        label: "judge_evidence",
                        detail: evidenceDecision.reason,
                        timestamp: Date.now(),
                    });

                    const evidenceCounts = countEvidence({ evidence });
                    const hasReusableMessageEvidence = evidenceCounts.messageEvidenceCount > 0;

                    if (evidenceDecision.sufficient && (toolHistory.length > 0 || hasReusableMessageEvidence)) {
                        confidence = evidenceDecision.confidence;
                        stopReason = "evidence_sufficient";
                        break;
                    }

                    if (toolHistory.length >= state.constraints.maxToolCalls) {
                        stopReason = "budget_exhausted";
                        traceEvents.push({
                            label: "stop",
                            detail: "Reached the tool-call budget.",
                            timestamp: Date.now(),
                        });
                        break;
                    }

                    if (Date.now() - startedAt >= state.constraints.maxLatencyBudgetMs) {
                        stopReason = "budget_exhausted";
                        traceEvents.push({
                            label: "stop",
                            detail: "Reached the latency budget.",
                            timestamp: Date.now(),
                        });
                        break;
                    }

                    const step = await planNextStep({
                        question: state.question,
                        goal: state.goal,
                        successCriteria: state.successCriteria,
                        confidence,
                        toolHistory,
                        candidateCapabilities: state.candidateCapabilities,
                        actorId: state.actorId,
                        replyContext: state.replyContext,
                        activeMemberTarget,
                        activeChannelTarget,
                        activeResolvedChannelIds,
                        activeRetrievalSession,
                        turnIntent: state.turnIntent,
                    });

                    if (!step.nextCapability) {
                        stopReason = "no_useful_next_step";
                        traceEvents.push({
                            label: "step",
                            detail: step.reason,
                            timestamp: Date.now(),
                        });
                        break;
                    }

                    const tool = step.nextCapability;
                    const signature = argsSignature(tool, step.arguments);
                    const seen = (repeated.get(signature) || 0) + 1;
                    repeated.set(signature, seen);
                    if (seen > state.constraints.maxRepeatedCallSignature) {
                        const warningRecord: ToolInvocationRecord = {
                            tool,
                            arguments: step.arguments,
                            summary: `Blocked: identical call already made. Try different arguments or a different capability.`,
                            learned: `This exact call (${signature}) was already made and blocked. You must use different arguments or choose a different capability.`,
                            confidenceImproved: false,
                            output: {
                                tool,
                                summary: `Repeated call blocked.`,
                                data: null,
                                errorMessage: `You already called ${tool} with these exact arguments. Change the arguments or choose a different capability.`,
                            },
                            durationMs: 0,
                            blocked: true,
                        };
                        toolHistory.push(warningRecord);
                        traceEvents.push({
                            label: "step",
                            detail: `Blocked repeated call ${signature}. Model warned to try different args.`,
                            timestamp: Date.now(),
                        });
                        const repeatedViolations = [...repeated.values()].filter((v) => v > state.constraints.maxRepeatedCallSignature).length;
                        if (repeatedViolations >= 2) {
                            stopReason = "confidence_plateau";
                            break;
                        }
                        continue;
                    }

                    const input = this.requestContext.get(state.requestId);
                    await input?.debugSession?.setPlanning(pass + 1);
                    await input?.debugSession?.setToolRunning(tool, [step.reason, step.learnedExpectation]);
                    const toolStartedAt = Date.now();
                    const output = await executeCapability(
                        tool,
                        {
                            guild: input?.guild || null,
                            question: state.question,
                            currentChannelId: state.channelId,
                            onProgress: async (toolName, summary) => {
                                traceEvents.push({
                                    label: "tool_progress",
                                    detail: `${toolName}: ${summary}`,
                                    timestamp: Date.now(),
                                });
                                await input?.debugSession?.setToolProgress?.(toolName, summary);
                            },
                        },
                        step.arguments
                    );

                    const evidenceItems = extractEvidence(output);
                    const learned = evidenceItems.map((item) => item.content).join(" | ") || output.summary;
                    const retrieval = getRetrievalSummary(output);
                    const resolvedMemberTarget = extractResolvedMemberTarget(output);
                    const resolvedChannelTarget = extractResolvedChannelTarget(output);
                    const retrievalSession = extractActiveRetrievalSession(output);
                    const record: ToolInvocationRecord = {
                        tool,
                        arguments: step.arguments,
                        summary: output.summary,
                        learned,
                        confidenceImproved: evidenceItems.some((item) => item.strength !== "weak"),
                        output,
                        durationMs: Date.now() - toolStartedAt,
                        retrievalSummary: retrieval,
                    };

                    toolHistory.push(record);
                    evidence.push(...evidenceItems);
                    if (retrieval) {
                        retrievalSummary = retrieval;
                    }
                    if (resolvedMemberTarget) {
                        activeMemberTarget = resolvedMemberTarget;
                    }
                    if (resolvedChannelTarget) {
                        activeChannelTarget = resolvedChannelTarget;
                        activeResolvedChannelIds = resolvedChannelTarget.resolvedIds;
                    } else if (retrieval?.searchedChannelIds?.length) {
                        activeResolvedChannelIds = retrieval.searchedChannelIds;
                    }
                    if (retrievalSession) {
                        activeRetrievalSession = mergeActiveRetrievalSession(
                            activeRetrievalSession,
                            retrievalSession
                        );
                    }
                    traceEvents.push(
                        { label: "step", detail: step.reason, timestamp: Date.now() },
                        {
                            label: "tool_result",
                            detail: `${tool}: ${output.summary}`,
                            timestamp: Date.now(),
                        }
                    );
                    if (tool === "retrieve_messages") {
                        const diagnostics =
                            output.data && typeof output.data === "object"
                                ? (output.data as Record<string, unknown>).retrievalDiagnostics
                                : null;
                        if (diagnostics && typeof diagnostics === "object") {
                            const scopedEmptyRetryAttempted = Boolean(
                                (diagnostics as Record<string, unknown>).scopedEmptyRetryAttempted
                            );
                            const scopedEmptyRetryRecovered = Boolean(
                                (diagnostics as Record<string, unknown>).scopedEmptyRetryRecovered
                            );
                            const retryStrategy = String(
                                (diagnostics as Record<string, unknown>).retryStrategy || "none"
                            );
                            traceEvents.push({
                                label: "retrieval_diagnostics",
                                detail:
                                    `continuationInput=${
                                        step.arguments.cursor || step.arguments.excludedMessageIds
                                            ? "yes"
                                            : "no"
                                    }; retryAttempted=${scopedEmptyRetryAttempted ? "yes" : "no"}; ` +
                                    `retryRecovered=${scopedEmptyRetryRecovered ? "yes" : "no"}; strategy=${retryStrategy}`,
                                timestamp: Date.now(),
                            });
                        }
                    }

                    await DiscordMemoryService.recordToolRun(
                        state.requestId,
                        state.guildId,
                        state.channelId,
                        state.actorId,
                        state.question,
                        tool,
                        JSON.stringify(step.arguments),
                        output.summary,
                        learned,
                        JSON.stringify(output),
                        record.confidenceImproved,
                        record.durationMs
                    );
                    await input?.debugSession?.setRetrievalSummary?.(retrieval || {
                        mode: "history",
                        cacheHit: false,
                        liveEscalated: false,
                        searchedChannelIds: [],
                        fetchedChannelIds: [],
                        cacheEnriched: false,
                        evidenceSufficient: false,
                        strongResultCount: 0,
                        weakResultCount: 0,
                        historyMessageCount: 0,
                        semanticMatchCount: 0,
                        accumulatedUniqueCount: 0,
                        sourceOrigin: "none",
                        continuationAvailable: false,
                        historyContinuationAvailable: false,
                        historyCursorByChannel: {},
                        semanticContinuationAvailable: false,
                        semanticCursor: null,
                        exhaustedChannelIds: [],
                        historyExhausted: false,
                        semanticExhausted: false,
                        beforeTimestamp: null,
                        afterTimestamp: null,
                        activeChannelIds: [],
                    });
                }

                const finalEvidenceDecision = await judgeEvidence({
                    question: state.question,
                    toolHistory,
                    evidence,
                    actorId: state.actorId,
                    candidateCapabilities: state.candidateCapabilities,
                });
                if (finalEvidenceDecision.sufficient) {
                    stopReason = "evidence_sufficient";
                } else if (stopReason === "evidence_sufficient") {
                    stopReason = "insufficient_evidence";
                }
                confidence = finalEvidenceDecision.confidence;
                traceEvents.push({
                    label: "judge_evidence",
                    detail: finalEvidenceDecision.reason,
                    timestamp: Date.now(),
                });

                const researchInput = this.requestContext.get(state.requestId);
                await researchInput?.debugSession?.setContextPreview?.({
                    recentChannelMessages: state.channelContext.map((m) => `${m.authorName}: ${m.content}`),
                    evidencePreview: evidence.slice(0, 6).map((e) => {
                        const channelLabel = e.channelName ? `#${e.channelName}` : "?";
                        const authorLabel = e.authorName
                            ? e.authorUsername && e.authorUsername !== e.authorName
                                ? `${e.authorName} (@${e.authorUsername})`
                                : e.authorName
                            : "?";
                        return `[${e.tool}] ${channelLabel} · ${authorLabel}: ${e.content}`;
                    }),
                    recentTurns: state.recentTurns.map((t) => `Q: ${t.question} | A: ${t.answer}`),
                });

                return {
                    toolHistory,
                    evidence,
                    activeMemberTarget,
                    activeChannelTarget,
                    activeResolvedChannelIds,
                    activeRetrievalSession,
                    retrievalSummary,
                    stopReason,
                    confidence,
                    traceEvents,
                };
            })
            .addNode("synthesize_answer", async (state: RuntimeState) => {
                const evidenceCounts = countEvidence(state);
                const shouldForceInsufficientGuard =
                    state.mode === "research" &&
                    state.confidence === "insufficient" &&
                    evidenceCounts.strongMessageEvidenceCount === 0 &&
                    evidenceCounts.liveEvidenceCount < 2;

                if (shouldForceInsufficientGuard) {
                    const responseDraft = buildConversationalRecovery({
                        question: state.question,
                        confidence: "insufficient",
                        evidence: state.evidence,
                        replyContext: state.replyContext,
                        priorTurns: state.recentTurns,
                        stopReason: state.stopReason,
                    });

                    return {
                        responseDraft,
                        confidence: "insufficient" as GroundedAnswerMode,
                        stopReason: state.stopReason || "insufficient_evidence",
                        traceEvents: appendTrace(
                            state,
                            "synthesize_answer",
                            "confidence=insufficient; web=off; enforced=no-guess guard"
                        ),
                    };
                }

                const effectiveConfidence =
                    state.mode === "research" && state.confidence === "insufficient"
                        ? answerConfidenceForInsufficient(state)
                        : state.confidence;
                const webDecision = decideConversationWebMode({
                    question: state.question,
                    classification: state.classification || classify(state.mode || "conversation"),
                    trigger: state.trigger,
                    conversationWebMode: state.requestedWebMode,
                    groundedAnswerMode: effectiveConfidence,
                });
                const webMode = state.mode === "conversation" ? webDecision.webMode : "off";
                let responseDraft = "";

                try {
                    const generated = await ModelGateway.generateText(
                        [
                            { role: "system", content: `${PromptRegistry.load("system/base")}\n\n${PromptRegistry.load("system/personality")}` },
                            {
                                role: "user",
                                content: PromptRegistry.render("runtime/synthesize_answer", {
                                    question: state.question,
                                    mode: state.mode || "conversation",
                                    confidence: effectiveConfidence,
                                    requester_display_name: state.requesterDisplayName,
                                    reply_context: state.replyContext?.content || "",
                                    recent_turns: summarizeRecentTurns(state.recentTurns),
                                    channel_context: formatChannelContext(state.channelContext),
                                    evidence: summarizeEvidence(state),
                                    stop_reason: state.stopReason || "",
                                    stop_detail:
                                        deriveStopDetail(state.stopReason, state.traceEvents || []) || "",
                                    continuation_available:
                                        state.activeRetrievalSession?.continuationAvailable
                                            ? "yes"
                                            : "no",
                                    active_retrieval_session: state.activeRetrievalSession
                                        ? `${state.activeRetrievalSession.mode} :: ${state.activeRetrievalSession.channelIds.join(", ")} :: unique=${state.activeRetrievalSession.accumulatedUniqueCount}`
                                        : "none",
                                }),
                            },
                        ],
                        {
                            webMode,
                            traceContext: {
                                traceLabel: "runtime_synthesize_answer",
                                questionPreview: state.question,
                                webContext: webDecision.reason,
                            },
                        }
                    );
                    responseDraft = sanitizeConversationalAnswer(generated);
                } catch {
                    responseDraft = "";
                }

                if (!responseDraft) {
                    responseDraft = buildFallbackAnswer({
                        question: state.question,
                        mode: state.mode,
                        confidence: effectiveConfidence,
                        evidence: state.evidence,
                        replyContext: state.replyContext,
                        recentTurns: state.recentTurns,
                        stopReason: state.stopReason,
                    });
                }

                responseDraft = sanitizeConversationalAnswer(responseDraft) || buildFallbackAnswer({
                    question: state.question,
                    mode: state.mode,
                    confidence: effectiveConfidence,
                    evidence: state.evidence,
                    replyContext: state.replyContext,
                    recentTurns: state.recentTurns,
                    stopReason: state.stopReason,
                });

                return {
                    responseDraft,
                    confidence: effectiveConfidence,
                    stopReason:
                        state.stopReason ||
                        (state.mode === "conversation" ? "direct_answer" : "insufficient_evidence"),
                    traceEvents: appendTrace(
                        state,
                        "synthesize_answer",
                        `confidence=${effectiveConfidence}; web=${webMode}`
                    ),
                };
            })
            .addNode("persist_run", async (state: RuntimeState) => {
                await DiscordMemoryService.recordRuntimeRun({
                    requestId: state.requestId,
                    threadId: state.threadId,
                    guildId: state.guildId,
                    channelId: state.channelId,
                    actorId: state.actorId,
                    trigger: state.trigger,
                    classificationMode: state.classification?.mode || "direct_answer",
                    runtimeMode: state.mode || "conversation",
                    stopReason: state.stopReason || "direct_answer",
                    confidence: state.confidence,
                    question: state.question,
                    answer: state.responseDraft || "",
                    traceEvents: state.traceEvents,
                });

                return {
                    traceEvents: appendTrace(state, "persist_run", "Saved runtime run and trace events."),
                };
            })
            .addEdge(START, "ingest_turn")
            .addEdge("ingest_turn", "load_context")
            .addEdge("load_context", "load_checkpoint")
            .addEdge("load_checkpoint", "load_memory")
            .addEdge("load_memory", "plan_turn")
            .addEdge("plan_turn", "route_mode")
            .addConditionalEdges("route_mode", (state: RuntimeState) =>
                state.mode === "research" ? "run_research_loop" : "synthesize_answer"
            )
            .addEdge("run_research_loop", "synthesize_answer")
            .addEdge("synthesize_answer", "persist_run")
            .addEdge("persist_run", END)
            .compile({ checkpointer });
    }

    public static async answer(input: TurnInput): Promise<RuntimeAnswer> {
        const graph = await this.getGraph();
        const config = getAppConfig();
        const requestId = randomUUID();
        const initialState: RuntimeState = {
            requestId,
            threadId: input.conversation.key,
            guildId: input.guild?.id || null,
            channelId: input.currentChannelId || null,
            actorId: input.user.id,
            requesterDisplayName: input.requesterDisplayName,
            trigger: input.trigger,
            conversationKind: input.conversation.kind,
            question: input.question,
            replyContext: input.replyContext || null,
            recentTurns: [],
            channelContext: [],
            permissionContext: {
                isAdmin: true,
                canReadChannel: true,
                canReadHistory: true,
                canSendMessages: true,
            },
            requestedWebMode: input.requestedWebMode || "off",
            mode: null,
            classification: null,
            goal: input.question,
            successCriteria: "Answer the question clearly.",
            candidateCapabilities: [],
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            activeRetrievalSession: null,
            toolHistory: [],
            evidence: [],
            retrievalSummary: null,
            turnIntent: null,
            stopReason: null,
            confidence: "insufficient",
            responseDraft: null,
            traceEvents: [],
            constraints: {
                maxToolCalls: config.runtime.maxToolCalls,
                maxResearchPasses: config.runtime.maxResearchPasses,
                maxRepeatedCallSignature: config.runtime.maxRepeatedCallSignature,
                maxLatencyBudgetMs: config.runtime.maxLatencyBudgetMs,
                maxPriorTurns: config.runtime.maxPriorTurns,
                maxChannelMessages: config.runtime.maxChannelMessages,
                maxToolRunsContext: config.runtime.maxToolRunsContext,
                maxEvidenceSlice: config.runtime.maxEvidenceSlice,
            },
        };

        try {
            this.requestContext.set(requestId, input);
            await input.debugSession?.setClassifying();
            await input.debugSession?.setRequesterContext?.(input.requesterDisplayName, input.trigger);
            await input.debugSession?.setConversationContext?.({
                threadId: initialState.threadId,
                kind: input.conversation.kind,
                replyAnchorMessageId: input.conversation.replyAnchorMessageId,
                replyContext: input.replyContext,
            });
            await input.debugSession?.setCheckpointThread?.(initialState.threadId);
            await input.debugSession?.setWebStatus?.(
                input.requestedWebMode === "auto" ? "enabled" : "off"
            );

            const result = await graph.invoke(initialState, {
                configurable: { thread_id: initialState.threadId, checkpoint_ns: "" },
            });

            const summary = countEvidence(result);
            await input.debugSession?.setClassification(
                result.classification?.mode || "direct_answer"
            );
            await input.debugSession?.setRuntimeMode?.(result.mode || "conversation");
            for (let index = 0; index < result.toolHistory.length; index += 1) {
                const tool = result.toolHistory[index] as ToolInvocationRecord;
                const itemCount =
                    tool.tool === "retrieve_messages" &&
                    tool.output.data &&
                    typeof tool.output.data === "object" &&
                    Array.isArray((tool.output.data as { combinedResults?: unknown[] }).combinedResults)
                        ? (tool.output.data as { combinedResults?: unknown[] }).combinedResults?.length
                        : undefined;
                await input.debugSession?.setPlanning(index + 1);
                await input.debugSession?.setToolRunning(tool.tool, [tool.learned]);
                await input.debugSession?.setToolResult(tool.tool, tool.summary, itemCount);
            }
            await input.debugSession?.setRetrievalSummary?.(
                result.retrievalSummary || {
                cacheHit: false,
                mode: "history",
                liveEscalated: false,
                searchedChannelIds: [],
                fetchedChannelIds: [],
                cacheEnriched: false,
                evidenceSufficient: false,
                strongResultCount: 0,
                weakResultCount: 0,
                historyMessageCount: 0,
                semanticMatchCount: 0,
                accumulatedUniqueCount: 0,
                sourceOrigin: "none",
                continuationAvailable: false,
                historyContinuationAvailable: false,
                historyCursorByChannel: {},
                semanticContinuationAvailable: false,
                semanticCursor: null,
                exhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: false,
                beforeTimestamp: null,
                afterTimestamp: null,
                activeChannelIds: [],
            }
            );
            await input.debugSession?.setGroundingSummary(
                {
                    messageEvidenceCount: summary.messageEvidenceCount,
                    liveEvidenceCount: summary.liveEvidenceCount,
                    sufficient: result.stopReason === "evidence_sufficient",
                },
                "judge",
                result.confidence
            );
            await input.debugSession?.setStopReason?.(
                result.stopReason || "direct_answer",
                deriveStopDetail(result.stopReason || "direct_answer", result.traceEvents || [])
            );
            await input.debugSession?.setConfidence?.(result.confidence);
            for (const event of result.traceEvents.slice(-8)) {
                await input.debugSession?.setTraceEvent?.(event.label, event.detail);
            }
            await input.debugSession?.setGenerating();
            await input.debugSession?.finishSuccess(result.stopReason || "completed");

            return {
                requestId,
                threadId: result.threadId,
                answer: result.responseDraft || buildDirectConversationFallback(input.question),
                citations: collectCitations(result.toolHistory as ToolInvocationRecord[]),
                classification: result.classification || classify(result.mode || "conversation"),
                toolRuns: (result.toolHistory as ToolInvocationRecord[]).filter((item) => !item.blocked).map((item) => item.output),
                confidence: result.confidence,
            };
        } catch (error) {
            await input.debugSession?.setTraceEvent?.(
                "runtime_error_fallback",
                "The runtime hit an internal error and returned a conversational fallback instead."
            );
            await input.debugSession?.finishError(error);
            return {
                requestId,
                threadId: initialState.threadId,
                answer: buildFallbackAnswer({
                    question: input.question,
                    mode: "conversation",
                    confidence: "best_effort",
                    evidence: [],
                    replyContext: input.replyContext || null,
                    recentTurns: [],
                    stopReason: null,
                }),
                citations: [],
                classification: classify("conversation"),
                toolRuns: [],
                confidence: "best_effort",
            };
        } finally {
            this.requestContext.delete(requestId);
        }
    }
}

function deriveStopDetail(
    stopReason: StopReason | null | undefined,
    traceEvents: Array<{ label: string; detail: string }>
): string | null {
    if (!stopReason) {
        return null;
    }

    const findLast = (predicate: (event: { label: string; detail: string }) => boolean) => {
        for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
            const event = traceEvents[index];
            if (predicate(event)) {
                return event.detail;
            }
        }
        return null;
    };

    if (stopReason === "budget_exhausted") {
        return findLast((event) => event.label === "stop");
    }

    if (stopReason === "confidence_plateau") {
        return findLast(
            (event) => event.label === "step" && event.detail.startsWith("Blocked repeated call ")
        );
    }

    if (stopReason === "no_useful_next_step") {
        return findLast((event) => event.label === "step");
    }

    if (stopReason === "evidence_sufficient" || stopReason === "insufficient_evidence") {
        return findLast((event) => event.label === "judge_evidence");
    }

    if (stopReason === "direct_answer") {
        return "Answered directly without entering the research loop.";
    }

    return null;
}

function collectCitations(toolHistory: ToolInvocationRecord[]) {
    const citations = new Map<string, { label: string; jumpLink: string }>();
    for (const item of toolHistory) {
        if (item.tool !== "retrieve_messages" || !item.output.data || typeof item.output.data !== "object") {
            continue;
        }
        const rows =
            (item.output.data as { combinedResults?: Array<Record<string, unknown>> }).combinedResults ||
            [];
        for (const row of rows) {
            if (typeof row.jumpLink !== "string" || !row.jumpLink) {
                continue;
            }
            if (!citations.has(row.jumpLink)) {
                citations.set(row.jumpLink, {
                    label: row.channelName ? `#${String(row.channelName)}` : item.tool,
                    jumpLink: row.jumpLink,
                });
            }
        }
    }
    return [...citations.values()].slice(0, 6);
}

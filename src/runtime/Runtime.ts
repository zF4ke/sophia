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
    EvidenceItem,
    GraphState,
    RetrievalSummary,
    RuntimeAnswer,
    RuntimeMode,
    RuntimeTraceEvent,
    StopReason,
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
    toolHistory: Annotation<ToolInvocationRecord[]>,
    evidence: Annotation<GraphState["evidence"]>,
    retrievalSummary: Annotation<GraphState["retrievalSummary"]>,
    stopReason: Annotation<StopReason | null>,
    confidence: Annotation<GroundedAnswerMode>,
    responseDraft: Annotation<string | null>,
    traceEvents: Annotation<GraphState["traceEvents"]>,
    constraints: Annotation<GraphState["constraints"]>,
});

type RuntimeState = typeof State.State;

type RetrievalPayload = {
    results?: Array<Record<string, unknown>>;
    sourceOrigin?: RetrievalSummary["sourceOrigin"];
    cacheHit?: boolean;
    liveEscalated?: boolean;
    searchedChannelIds?: string[];
    fetchedChannelIds?: string[];
    cacheEnriched?: boolean;
    evidenceSufficient?: boolean;
    strongResultCount?: number;
    weakResultCount?: number;
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
        displayName: String(item.displayName || "Unknown"),
        username: String(item.username || "unknown"),
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

function argsSignature(tool: DiscordToolName, args: Record<string, string | number | undefined>) {
    return `${tool}:${JSON.stringify(args)}`;
}

function getRetrievalSummary(run: DiscordToolResult): RetrievalSummary | null {
    if (run.tool !== "retrieve_messages" || !run.data || typeof run.data !== "object") {
        return null;
    }

    const payload = run.data as RetrievalPayload;
    return {
        cacheHit: Boolean(payload.cacheHit),
        liveEscalated: Boolean(payload.liveEscalated),
        searchedChannelIds: (payload.searchedChannelIds || []).map(String),
        fetchedChannelIds: (payload.fetchedChannelIds || []).map(String),
        cacheEnriched: Boolean(payload.cacheEnriched),
        evidenceSufficient: Boolean(payload.evidenceSufficient),
        strongResultCount: Number(payload.strongResultCount || 0),
        weakResultCount: Number(payload.weakResultCount || 0),
        sourceOrigin: (payload.sourceOrigin || "none") as RetrievalSummary["sourceOrigin"],
    };
}

function extractEvidence(run: DiscordToolResult): EvidenceItem[] {
    if (run.tool === "retrieve_messages" && run.data && typeof run.data === "object") {
        const payload = run.data as RetrievalPayload;
        const rows = payload.results || [];
        const sourceOrigin = (payload.sourceOrigin || "none") as RetrievalSummary["sourceOrigin"];
        return rows.slice(0, 4).map((item) => {
            const content = String(item.content || "").slice(0, 260);
            const lexicalScore = Number(item.lexicalScore || 0);
            const strength =
                lexicalScore >= 2 || (lexicalScore >= 1 && content.length >= 80)
                    ? "strong"
                    : "weak";

            return {
                tool: "retrieve_messages",
                summary: run.summary,
                content,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.retrieve_messages,
                strength,
                sourceOrigin,
                authorId: item.authorId == null ? null : String(item.authorId),
                authorName: item.authorName == null ? null : String(item.authorName),
                channelId: item.channelId == null ? null : String(item.channelId),
                channelName: item.channelName == null ? null : String(item.channelName),
                jumpLink: item.jumpLink == null ? null : String(item.jumpLink),
                createdTimestamp: item.createdTimestamp == null ? null : Number(item.createdTimestamp),
            };
        });
    }

    if (run.tool === "resolve_member_identity" && run.data) {
        const item = run.data as Record<string, unknown>;
        const currentState =
            item.isCurrentGuildMember === false ? "historical guild memory" : "current guild";
        return [
            {
                tool: "resolve_member_identity",
                summary: run.summary,
                content: `${String(item.displayName || "Unknown")} (@${String(item.username || "unknown")}) from ${currentState}`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.resolve_member_identity,
                strength: "metadata",
                sourceOrigin: "none",
                authorId:
                    item.resolvedId == null
                        ? item.id == null
                            ? null
                            : String(item.id)
                        : String(item.resolvedId),
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
        return [
            {
                tool: "get_member_profile",
                summary: run.summary,
                content: `${String(item.displayName || "Unknown")} (@${String(item.username || "unknown")}) roles=${roles || "none"}`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.get_member_profile,
                strength: "metadata",
                sourceOrigin: "none",
                authorId: item.id == null ? null : String(item.id),
                authorName: item.displayName == null ? null : String(item.displayName),
            },
        ];
    }

    if (run.tool === "list_members" && run.data && typeof run.data === "object") {
        const members = (run.data as { members?: Array<Record<string, unknown>> }).members || [];
        return members.slice(0, 4).map((item) => ({
            tool: "list_members",
            summary: run.summary,
            content: `${String(item.displayName || "Unknown")} (@${String(item.username || "unknown")})`,
            evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_members,
            strength: "metadata",
            sourceOrigin: "none",
            authorId: item.id == null ? null : String(item.id),
            authorName: item.displayName == null ? null : String(item.displayName),
        }));
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

function buildFallbackAnswer(state: Pick<RuntimeState, "question" | "mode" | "confidence" | "evidence" | "replyContext" | "recentTurns">): string {
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
        args: Record<string, string | number | undefined>
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
                    3
                );
                const channelMessages = state.channelId
                    ? await DiscordMemoryService.getRecentChannelMessagesAsync(state.channelId, 10)
                    : [];
                const channelContext = channelMessages.map((msg) => ({
                    authorName: msg.authorName,
                    content: msg.content.slice(0, 150),
                    createdTimestamp: msg.createdTimestamp,
                }));
                const input = this.requestContext.get(state.requestId);
                await input?.debugSession?.setContextPreview?.({
                    recentChannelMessages: channelContext.map((m) => `${m.authorName}: ${m.content}`),
                    evidencePreview: [],
                    recentTurns: recentTurns.map((t) => `Q: ${t.question} | A: ${t.answer}`),
                });
                return {
                    recentTurns,
                    channelContext,
                    traceEvents: appendTrace(
                        state,
                        "load_memory",
                        `Loaded ${recentTurns.length} prior conversation turn(s) and ${channelContext.length} recent channel message(s).`
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
                    }
                );

                return {
                    mode: plan.mode,
                    classification: classify(plan.mode),
                    goal: plan.goal,
                    successCriteria: plan.successCriteria,
                    candidateCapabilities: plan.candidateCapabilities,
                    confidence: plan.confidence,
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

                    if (evidenceDecision.sufficient && toolHistory.length > 0) {
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
                        stopReason = "confidence_plateau";
                        traceEvents.push({
                            label: "step",
                            detail: `Blocked repeated call ${signature}.`,
                            timestamp: Date.now(),
                        });
                        break;
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
                    traceEvents.push(
                        { label: "step", detail: step.reason, timestamp: Date.now() },
                        {
                            label: "tool_result",
                            detail: `${tool}: ${output.summary}`,
                            timestamp: Date.now(),
                        }
                    );

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
                        record.confidenceImproved,
                        record.durationMs
                    );
                    await input?.debugSession?.setRetrievalSummary?.(retrieval || {
                        cacheHit: false,
                        liveEscalated: false,
                        searchedChannelIds: [],
                        fetchedChannelIds: [],
                        cacheEnriched: false,
                        evidenceSufficient: false,
                        strongResultCount: 0,
                        weakResultCount: 0,
                        sourceOrigin: "none",
                    });
                }

                const finalEvidenceDecision = await judgeEvidence({
                    question: state.question,
                    toolHistory,
                    evidence,
                    actorId: state.actorId,
                    candidateCapabilities: state.candidateCapabilities,
                });
                if (stopReason === "insufficient_evidence" && finalEvidenceDecision.sufficient) {
                    stopReason = "evidence_sufficient";
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
                    evidencePreview: evidence.slice(0, 4).map((e) => `[${e.tool}] ${e.authorName || "?"}: ${e.content}`),
                    recentTurns: state.recentTurns.map((t) => `Q: ${t.question} | A: ${t.answer}`),
                });

                return {
                    toolHistory,
                    evidence,
                    activeMemberTarget,
                    activeChannelTarget,
                    activeResolvedChannelIds,
                    retrievalSummary,
                    stopReason,
                    confidence,
                    traceEvents,
                };
            })
            .addNode("synthesize_answer", async (state: RuntimeState) => {
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
                    });
                }

                responseDraft = sanitizeConversationalAnswer(responseDraft) || buildFallbackAnswer({
                    question: state.question,
                    mode: state.mode,
                    confidence: effectiveConfidence,
                    evidence: state.evidence,
                    replyContext: state.replyContext,
                    recentTurns: state.recentTurns,
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
            toolHistory: [],
            evidence: [],
            retrievalSummary: null,
            stopReason: null,
            confidence: "insufficient",
            responseDraft: null,
            traceEvents: [],
            constraints: {
                maxToolCalls: config.runtime.maxToolCalls,
                maxResearchPasses: config.runtime.maxResearchPasses,
                maxRepeatedCallSignature: config.runtime.maxRepeatedCallSignature,
                maxLatencyBudgetMs: config.runtime.maxLatencyBudgetMs,
                maxCostTier: config.runtime.maxCostTier,
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
                    Array.isArray((tool.output.data as { results?: unknown[] }).results)
                        ? (tool.output.data as { results?: unknown[] }).results?.length
                        : undefined;
                await input.debugSession?.setPlanning(index + 1);
                await input.debugSession?.setToolRunning(tool.tool, [tool.learned]);
                await input.debugSession?.setToolResult(tool.tool, tool.summary, itemCount);
            }
            await input.debugSession?.setRetrievalSummary?.(
                result.retrievalSummary || {
                    cacheHit: false,
                    liveEscalated: false,
                    searchedChannelIds: [],
                    fetchedChannelIds: [],
                    cacheEnriched: false,
                    evidenceSufficient: false,
                    strongResultCount: 0,
                    weakResultCount: 0,
                    sourceOrigin: "none",
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
            await input.debugSession?.setStopReason?.(result.stopReason || "direct_answer");
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
                toolRuns: (result.toolHistory as ToolInvocationRecord[]).map((item) => item.output),
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

function collectCitations(toolHistory: ToolInvocationRecord[]) {
    const citations = new Map<string, { label: string; jumpLink: string }>();
    for (const item of toolHistory) {
        if (item.tool !== "retrieve_messages" || !item.output.data || typeof item.output.data !== "object") {
            continue;
        }
        const rows = (item.output.data as { results?: Array<Record<string, unknown>> }).results || [];
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

import { ModelGateway } from "@/ai/ModelGateway";
import { CapabilityRegistry } from "@/discord/capabilities/CapabilityRegistry";
import {
    extractDeterministicIntent,
    mergeIntent,
    normalize,
    parseModelIntent,
} from "@/runtime/intentExtraction";
import { PromptRegistry } from "@/runtime/PromptRegistry";
import type {
    ActiveRetrievalSession,
    ChannelContextMessage,
    EvidenceDecision,
    GraphState,
    PlanDecision,
    RuntimeMode,
    StepDecision,
    TurnIntent,
    ToolArguments,
    ToolArgumentValue,
    ToolInvocationRecord,
    TurnInput,
} from "@/runtime/contracts";
import type { GroundedAnswerMode, RequestClassification } from "@/shared/appTypes";
import {
    DISCORD_TOOL_NAMES,
    type DiscordToolName,
} from "@/shared/discordTools";

const VALID_TOOL_NAMES = new Set<DiscordToolName>(DISCORD_TOOL_NAMES);
const GENERIC_RESEARCH_CAPABILITIES: DiscordToolName[] = [
    "resolve_member_identity",
    "resolve_channel_targets",
    "retrieve_messages",
    "list_guild_structure",
];
const GENERIC_STEP_ORDER: DiscordToolName[] = [
    "retrieve_messages",
    "list_guild_structure",
    "get_member_profile",
    "get_guild_context",
    "list_members",
];

function hasExactSnowflake(text: string): boolean {
    return /(?:<[@#!]?(\d+)>|\b\d{6,25}\b)/.test(text);
}

function extractMemberMentionId(text: string): string | null {
    return text.match(/<@!?(\d+)>/)?.[1] || null;
}

function extractChannelMentionId(text: string): string | null {
    return text.match(/<#(\d+)>/)?.[1] || null;
}

function extractBareSnowflake(text: string): string | null {
    return text.match(/\b\d{6,25}\b/)?.[0] || null;
}

function extractNamedGuildTargetReference(text: string): string | null {
    const targetPatterns = [
        /(?:^|[\s([{'"`])#([\p{L}\p{N}][\p{L}\p{N}-]{1,63})/u,
        /(?:^|\b)(?:canal|channel|categoria|category)[\s\u200b-\u200d\u2060]*#?([\p{L}\p{N}][\p{L}\p{N}-]{1,63})/iu,
    ];

    for (const pattern of targetPatterns) {
        const match = text.match(pattern);
        const target = match?.[1]?.trim();
        if (target) {
            return target;
        }
    }

    return null;
}

function extractStructuralMemberReference(
    question: string,
    replyContext?: Pick<TurnInput, "replyContext">["replyContext"] | null
): string | null {
    const namedMemberMatch = question.match(
        /\b(?:o|a)?\s*([\p{L}\p{N}_.-]{3,32})\s+(?:mandou|enviou|postou|disse|falou|citou|mencionou)\b/iu
    )?.[1];

    if (namedMemberMatch) {
        return namedMemberMatch;
    }

    if (extractMemberMentionId(question)) {
        return extractMemberMentionId(question);
    }

    const bare = extractBareSnowflake(question);
    if (bare && !extractChannelMentionId(question)) {
        return bare;
    }

    return null;
}

function extractStructuralChannelReference(
    question: string,
    replyContext?: Pick<TurnInput, "replyContext">["replyContext"] | null
): string | null {
    return (
        extractChannelMentionId(question) ||
        extractNamedGuildTargetReference(question) ||
        (replyContext?.content ? extractChannelMentionId(replyContext.content) : null) ||
        (replyContext?.content ? extractNamedGuildTargetReference(replyContext.content) : null) ||
        null
    );
}

function isClearlyLocalReplyFollowUp(input: Pick<TurnInput, "question" | "replyContext">): boolean {
    if (!input.replyContext) {
        return false;
    }

    const compact = normalize(input.question);
    if (!compact || compact.length > 80) {
        return false;
    }

    return !/[<#@]/.test(input.question) && !hasExactSnowflake(input.question);
}

function sanitizeCandidateCapabilities(value: unknown): DiscordToolName[] {
    if (!Array.isArray(value)) {
        return [];
    }

    const sanitized = value.filter((item): item is DiscordToolName =>
        typeof item === "string" && VALID_TOOL_NAMES.has(item as DiscordToolName)
    );

    return [...new Set(sanitized)];
}

function mergeUniqueCapabilities(
    base: DiscordToolName[],
    extra: DiscordToolName[]
): DiscordToolName[] {
    return [...new Set([...base, ...extra])];
}

function getStructuralCapabilityHints(
    input: Pick<TurnInput, "question" | "replyContext" | "guild">
): DiscordToolName[] {
    if (!input.guild) {
        return [];
    }

    const hinted: DiscordToolName[] = [];
    const memberMention = extractMemberMentionId(input.question);
    const channelReference = extractStructuralChannelReference(
        input.question,
        input.replyContext
    );
    const bareSnowflake = extractBareSnowflake(input.question);

    if (memberMention) {
        hinted.push("resolve_member_identity");
    }

    if (channelReference) {
        hinted.push("resolve_channel_targets");
    }

    if (bareSnowflake && !memberMention && !channelReference) {
        hinted.push("resolve_member_identity", "resolve_channel_targets");
    }

    if (hinted.length) {
        hinted.push("retrieve_messages");
    }

    return mergeUniqueCapabilities([], hinted);
}

function ensureResearchCapabilities(capabilities: DiscordToolName[]): DiscordToolName[] {
    return capabilities.length ? capabilities : [...GENERIC_RESEARCH_CAPABILITIES];
}

function sanitizeReason(value: unknown, fallback: string): string {
    return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function sanitizeConfidence(value: unknown, fallback: GroundedAnswerMode): GroundedAnswerMode {
    return value === "confident" || value === "best_effort" || value === "insufficient"
        ? value
        : fallback;
}

function sanitizeToolArgumentValue(value: unknown): ToolArgumentValue | undefined {
    if (
        value == null ||
        typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean"
    ) {
        return value as ToolArgumentValue;
    }
    if (Array.isArray(value)) {
        return value
            .map((item) => sanitizeToolArgumentValue(item))
            .filter((item): item is ToolArgumentValue => item !== undefined);
    }
    if (typeof value === "object") {
        return Object.fromEntries(
            Object.entries(value as Record<string, unknown>)
                .map(([key, raw]) => [key, sanitizeToolArgumentValue(raw)])
                .filter(([, raw]) => raw !== undefined)
        ) as ToolArgumentValue;
    }
    return undefined;
}

function parseFlexibleTimestamp(value: unknown): number | undefined {
    if (typeof value === "number" && Number.isFinite(value)) {
        return value;
    }
    if (typeof value !== "string" || !value.trim()) {
        return undefined;
    }

    const parsed = Date.parse(value.trim());
    return Number.isNaN(parsed) ? undefined : parsed;
}

function sanitizeArguments(
    value: unknown
): ToolArguments {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return {};
    }

    return Object.fromEntries(
        Object.entries(value as Record<string, unknown>)
            .map(([key, raw]) => [key, sanitizeToolArgumentValue(raw)])
            .filter(([, raw]) => raw !== undefined)
    ) as ToolArguments;
}

function extractLatestRetrieval(toolHistory: ToolInvocationRecord[]) {
    for (let index = toolHistory.length - 1; index >= 0; index -= 1) {
        const item = toolHistory[index];
        if (item.tool === "retrieve_messages") {
            return item.retrievalSummary || null;
        }
    }

    return null;
}

function extractLatestMemberProfileId(toolHistory: ToolInvocationRecord[]): string | null {
    for (let index = toolHistory.length - 1; index >= 0; index -= 1) {
        const item = toolHistory[index];
        if (item.tool !== "get_member_profile" && item.tool !== "resolve_member_identity") {
            continue;
        }

        const data = item.output.data as Record<string, unknown> | null;
        if (!data || typeof data !== "object") {
            continue;
        }

        if (data.resolvedId != null) {
            return String(data.resolvedId);
        }

        return data.id == null ? null : String(data.id);
    }

    return null;
}

function extractLatestResolvedMember(toolHistory: ToolInvocationRecord[]) {
    for (let index = toolHistory.length - 1; index >= 0; index -= 1) {
        const item = toolHistory[index];
        if (item.tool !== "resolve_member_identity" && item.tool !== "get_member_profile") {
            continue;
        }

        const data = item.output.data as Record<string, unknown> | null;
        if (!data || typeof data !== "object") {
            continue;
        }

        return {
            resolvedId:
                data.resolvedId == null
                    ? data.id == null
                        ? null
                        : String(data.id)
                    : String(data.resolvedId),
            isCurrentGuildMember:
                data.isCurrentGuildMember == null ? true : Boolean(data.isCurrentGuildMember),
        };
    }

    return null;
}

function extractLatestResolvedChannelIds(toolHistory: ToolInvocationRecord[]): string[] {
    for (let index = toolHistory.length - 1; index >= 0; index -= 1) {
        const item = toolHistory[index];
        if (item.tool !== "resolve_channel_targets") {
            continue;
        }

        const data = item.output.data as Record<string, unknown> | null;
        if (!data || typeof data !== "object" || !Array.isArray(data.resolvedIds)) {
            continue;
        }

        return data.resolvedIds.map(String);
    }

    return [];
}

function extractLatestResolvedChannelTarget(toolHistory: ToolInvocationRecord[]) {
    for (let index = toolHistory.length - 1; index >= 0; index -= 1) {
        const item = toolHistory[index];
        if (item.tool !== "resolve_channel_targets" && item.tool !== "list_guild_structure") {
            continue;
        }

        const data = item.output.data as Record<string, unknown> | null;
        if (!data || typeof data !== "object") {
            continue;
        }

        if (item.tool === "resolve_channel_targets") {
            const resolvedIds = Array.isArray(data.resolvedIds) ? data.resolvedIds.map(String) : [];
            return {
                query: typeof data.query === "string" ? data.query : "",
                resolvedIds,
                entries: Array.isArray(data.entries) ? data.entries : [],
            };
        }

        const focusedResolvedIds = Array.isArray(data.focusedResolvedIds)
            ? data.focusedResolvedIds.map(String)
            : [];
        const focusedEntries = Array.isArray(data.focusedEntries) ? data.focusedEntries : [];
        const query = typeof data.query === "string" ? data.query : "";
        if (focusedResolvedIds.length || focusedEntries.length || query) {
            return {
                query,
                resolvedIds: focusedResolvedIds,
                entries: focusedEntries,
            };
        }
    }

    return null;
}

function extractAmbiguousMemberCandidate(toolHistory: ToolInvocationRecord[]): {
    displayName: string;
    identifiers: string[];
} | null {
    for (let index = toolHistory.length - 1; index >= 0; index -= 1) {
        const item = toolHistory[index];
        if (item.tool !== "list_members") {
            continue;
        }

        const data = item.output.data as Record<string, unknown> | null;
        const members = data && Array.isArray(data.members) ? data.members : [];
        if (!members.length) {
            continue;
        }

        const byName = new Map<string, Set<string>>();
        const displayByName = new Map<string, string>();
        const identifiersByName = new Map<string, string[]>();
        for (const raw of members) {
            if (!raw || typeof raw !== "object") {
                continue;
            }
            const member = raw as Record<string, unknown>;
            const displayName =
                typeof member.displayName === "string" && member.displayName.trim()
                    ? member.displayName.trim()
                    : typeof member.username === "string" && member.username.trim()
                      ? member.username.trim()
                      : "";
            if (!displayName) {
                continue;
            }
            const key = normalize(displayName);
            if (!key) {
                continue;
            }

            const id =
                member.id == null
                    ? member.userId == null
                        ? null
                        : String(member.userId)
                    : String(member.id);
            const username =
                typeof member.username === "string" && member.username.trim()
                    ? member.username.trim()
                    : null;
            const distinctMarker = id || username || displayName;
            const bestIdentifier = username || id || displayName;

            if (!byName.has(key)) {
                byName.set(key, new Set<string>());
                displayByName.set(key, displayName);
                identifiersByName.set(key, []);
            }
            byName.get(key)?.add(distinctMarker);
            identifiersByName.get(key)?.push(bestIdentifier);
        }

        for (const [key, variants] of byName.entries()) {
            if (variants.size > 1) {
                return {
                    displayName: displayByName.get(key) || key,
                    identifiers: identifiersByName.get(key) || [],
                };
            }
        }
    }

    return null;
}

function extractProfiledMemberIdentifiers(toolHistory: ToolInvocationRecord[]): Set<string> {
    const profiled = new Set<string>();
    for (const item of toolHistory) {
        if (item.tool !== "get_member_profile") {
            continue;
        }
        const data = item.output.data as Record<string, unknown> | null;
        if (!data || typeof data !== "object") {
            continue;
        }
        if (typeof data.username === "string" && data.username.trim()) {
            profiled.add(normalize(data.username.trim()));
        }
        if (data.id != null) {
            profiled.add(String(data.id));
        }
    }
    return profiled;
}

function hasToolRun(toolHistory: ToolInvocationRecord[], tool: DiscordToolName): boolean {
    return toolHistory.some((item) => item.tool === tool);
}

function classifyFallbackMode(input: TurnInput): RuntimeMode {
    if (!input.guild) {
        return "conversation";
    }

    if (isClearlyLocalReplyFollowUp(input)) {
        return "conversation";
    }

    return "research";
}

function shouldChainStructureCapabilities(capabilities: DiscordToolName[]): boolean {
    return (
        capabilities.includes("list_guild_structure") ||
        capabilities.includes("resolve_channel_targets")
    );
}

function normalizePlanDecision(input: TurnInput, plan: PlanDecision): PlanDecision {
    let mode: RuntimeMode = plan.mode === "research" ? "research" : "conversation";
    let candidateCapabilities = sanitizeCandidateCapabilities(plan.candidateCapabilities);
    const structuralCapabilities = getStructuralCapabilityHints(input);

    if (!input.guild) {
        mode = "conversation";
        candidateCapabilities = [];
    } else if (mode === "research") {
        candidateCapabilities = ensureResearchCapabilities(candidateCapabilities);
    } else {
        candidateCapabilities = [];
    }

    if (structuralCapabilities.length) {
        mode = "research";
        candidateCapabilities = ensureResearchCapabilities(
            mergeUniqueCapabilities(candidateCapabilities, structuralCapabilities)
        );
    }

    if (mode === "research" && shouldChainStructureCapabilities(candidateCapabilities)) {
        candidateCapabilities = mergeUniqueCapabilities(candidateCapabilities, [
            "resolve_channel_targets",
            "list_guild_structure",
            "retrieve_messages",
        ]);
    }

    if (plan.mode === "refusal") {
        mode = "conversation";
        candidateCapabilities = [];
    }

    if (mode !== "research" && isClearlyLocalReplyFollowUp(input)) {
        mode = "conversation";
        candidateCapabilities = [];
    }

    const confidence = sanitizeConfidence(
        plan.confidence,
        mode === "research" ? "best_effort" : "confident"
    );

    return {
        mode,
        reason:
            plan.mode === "refusal"
                ? "The runtime should stay conversational instead of refusing this turn."
                : sanitizeReason(
                      plan.reason,
                      mode === "research"
                          ? "The request may depend on current guild state or Discord history."
                          : "The request can stay conversational."
                  ),
        goal: sanitizeReason(plan.goal, input.question),
        successCriteria: sanitizeReason(
            plan.successCriteria,
            mode === "research"
                ? "Use current-guild discovery or retrieval only if it materially improves the answer."
                : "Reply naturally and keep the conversation moving."
        ),
        candidateCapabilities,
        confidence:
            mode === "conversation" && confidence === "insufficient"
                ? "best_effort"
                : confidence,
        intent: plan.intent,
    };
}

export function classify(mode: RuntimeMode): RequestClassification {
    return {
        mode: mode === "research" ? "discord_grounded" : "direct_answer",
        reason:
            mode === "research"
                ? "The runtime routed this request through Discord retrieval."
                : "The runtime can answer without Discord retrieval.",
    };
}

function summarizeToolHistory(toolHistory: ToolInvocationRecord[]): string {
    return toolHistory.length
        ? toolHistory
              .map(
                  (item, index) =>
                      `${index + 1}. ${item.tool} args=${JSON.stringify(item.arguments)} learned=${item.learned} summary=${item.summary}`
              )
              .join("\n")
        : "No tools have run yet.";
}

export function summarizeEvidence(state: Pick<GraphState, "evidence">): string {
    return state.evidence.length
        ? state.evidence
              .map((item) => {
                  const meta = [
                      item.channelName ? `#${item.channelName}` : null,
                      item.authorName || null,
                      item.sourceOrigin,
                      item.strength,
                  ]
                      .filter(Boolean)
                      .join(" · ");
                  const timestamp =
                      item.createdTimestamp != null
                          ? ` @${new Date(item.createdTimestamp).toISOString()}`
                          : "";
                  return `${item.tool}: ${meta ? `[${meta}] ` : ""}${item.content}${timestamp}`;
              })
              .join("\n")
        : "No Discord evidence collected.";
}

export function countEvidence(state: Pick<GraphState, "evidence">) {
    return state.evidence.reduce(
        (acc, item) => {
            if (
                item.evidenceRole === "message_evidence" ||
                item.evidenceRole === "history_evidence" ||
                item.evidenceRole === "semantic_evidence"
            ) {
                acc.messageEvidenceCount += 1;
                if (item.strength === "strong") {
                    acc.strongMessageEvidenceCount += 1;
                } else if (item.strength === "weak") {
                    acc.weakMessageEvidenceCount += 1;
                }
            }
            if (
                item.evidenceRole === "live_evidence" ||
                item.evidenceRole === "discovery_only"
            ) {
                acc.liveEvidenceCount += 1;
            }
            return acc;
        },
        {
            messageEvidenceCount: 0,
            strongMessageEvidenceCount: 0,
            weakMessageEvidenceCount: 0,
            liveEvidenceCount: 0,
        }
    );
}

export function guessPlan(
    input: TurnInput,
    activeRetrievalSession?: ActiveRetrievalSession | null
): PlanDecision {
    const mode = classifyFallbackMode(input);
    const intent = mergeIntent(
        extractDeterministicIntent(input.question, activeRetrievalSession),
        {}
    );
    return normalizePlanDecision(input, {
        mode,
        reason:
            mode === "research"
                ? "Use current-guild discovery and retrieval if it helps answer the turn."
                : "Keep this turn conversational by default.",
        goal: input.question,
        successCriteria:
            mode === "research"
                ? "Use the smallest set of current-guild tools needed for a grounded answer."
                : "Reply conversationally and directly.",
        candidateCapabilities:
            mode === "research" ? [...GENERIC_RESEARCH_CAPABILITIES] : [],
        confidence: mode === "research" ? "best_effort" : "confident",
        intent,
    });
}

export async function planWithModel(
    input: TurnInput,
    channelContext: ChannelContextMessage[] = [],
    recentTurns: Pick<GraphState, "recentTurns">["recentTurns"] = [],
    activeTargets?: Pick<
        GraphState,
        "activeMemberTarget" | "activeChannelTarget" | "activeResolvedChannelIds" | "activeRetrievalSession"
    >
): Promise<PlanDecision> {
    const activeSession = activeTargets?.activeRetrievalSession ?? null;
    const fallback = guessPlan(input, activeSession);
    const deterministicIntent = extractDeterministicIntent(input.question, activeSession);

    try {
        const raw = await ModelGateway.generateJson<PlanDecision>(
            [
                { role: "system", content: PromptRegistry.load("system/base") },
                {
                    role: "user",
                    content: PromptRegistry.render("runtime/plan_turn", {
                        trigger: input.trigger,
                        guild_available: input.guild ? "yes" : "no",
                        question: input.question,
                        reply_context: input.replyContext?.content || "",
                        recent_turns: recentTurns.length
                            ? recentTurns
                                  .map(
                                      (turn, index) =>
                                          `${index + 1}. user=${turn.question} | sophia=${turn.answer}`
                                  )
                                  .join("\n")
                            : input.replyContext
                              ? "reply context available"
                              : "none",
                        channel_context: formatChannelContext(channelContext),
                        active_member_target: activeTargets?.activeMemberTarget
                            ? `${activeTargets.activeMemberTarget.displayName} (@${activeTargets.activeMemberTarget.username})`
                            : "none",
                        active_channel_target: activeTargets?.activeChannelTarget
                            ? `${activeTargets.activeChannelTarget.query} -> ${activeTargets.activeResolvedChannelIds.join(", ") || "no resolved channel ids"}`
                            : "none",
                        active_retrieval_session: activeTargets?.activeRetrievalSession
                            ? `${activeTargets.activeRetrievalSession.mode} :: ${activeTargets.activeRetrievalSession.channelIds.join(", ")} :: continuation=${activeTargets.activeRetrievalSession.continuationAvailable ? "yes" : "no"}`
                            : "none",
                        capability_registry: CapabilityRegistry.describeForPrompt(),
                    }),
                },
            ],
            fallback,
            {
                traceContext: {
                    traceLabel: "runtime_plan_turn",
                    questionPreview: input.question,
                },
            }
        );
        const modelIntent = parseModelIntent(raw);
        const intent = mergeIntent(deterministicIntent, modelIntent);
        const plan = normalizePlanDecision(input, raw);
        return { ...plan, intent };
    } catch {
        return fallback;
    }
}

function normalizeEvidenceDecision(
    state: Pick<
        GraphState,
        "question" | "toolHistory" | "evidence" | "actorId" | "candidateCapabilities"
    >,
    decision: EvidenceDecision
): EvidenceDecision {
    const retrieval = extractLatestRetrieval(state.toolHistory);
    const memberProfileId = extractLatestMemberProfileId(state.toolHistory);
    const resolvedChannelTarget = extractLatestResolvedChannelTarget(state.toolHistory);
    const counts = countEvidence(state);
    const hasStructureRun = hasToolRun(state.toolHistory, "list_guild_structure");
    const hasRetrieveMessagesRun = hasToolRun(state.toolHistory, "retrieve_messages");

    if (memberProfileId === state.actorId && counts.messageEvidenceCount === 0) {
        return {
            sufficient: true,
            confidence: "confident",
            reason: "The requesting member was resolved directly.",
        };
    }

    if (
        counts.messageEvidenceCount === 0 &&
        resolvedChannelTarget &&
        state.candidateCapabilities.includes("list_guild_structure") &&
        !hasStructureRun
    ) {
        return {
            sufficient: false,
            confidence: "best_effort",
            reason: "The channel or category target was resolved, but the guild structure still needs inspection.",
        };
    }

    if (
        counts.messageEvidenceCount === 0 &&
        hasStructureRun &&
        resolvedChannelTarget?.resolvedIds.length &&
        state.candidateCapabilities.includes("retrieve_messages") &&
        !hasRetrieveMessagesRun
    ) {
        return {
            sufficient: false,
            confidence: "best_effort",
            reason: "The matched structure still needs scoped message retrieval before answering.",
        };
    }

    if (retrieval) {
        if (counts.messageEvidenceCount === 0) {
            return {
                sufficient: false,
                confidence: counts.liveEvidenceCount > 0 ? "best_effort" : "insufficient",
                reason: "Message retrieval did not produce usable message evidence yet.",
            };
        }

        return {
            sufficient: true,
            confidence:
                retrieval.strongResultCount > 0
                    ? sanitizeConfidence(decision.confidence, "best_effort")
                    : "best_effort",
            reason: sanitizeReason(
                decision.reason,
                "Discord message evidence is available."
            ),
        };
    }

    if (counts.messageEvidenceCount > 0) {
        return {
            sufficient: true,
            confidence: sanitizeConfidence(decision.confidence, "best_effort"),
            reason: sanitizeReason(decision.reason, "Discord message evidence is available."),
        };
    }

    if (counts.liveEvidenceCount > 0) {
        if (!decision.sufficient) {
            return {
                sufficient: false,
                confidence: sanitizeConfidence(decision.confidence, "best_effort"),
                reason: sanitizeReason(
                    decision.reason,
                    "Discord live metadata is available, but it is not enough yet."
                ),
            };
        }

        return {
            sufficient: true,
            confidence: sanitizeConfidence(decision.confidence, "best_effort"),
            reason: sanitizeReason(decision.reason, "Discord live metadata is available."),
        };
    }

    return {
        sufficient: false,
        confidence: "insufficient",
        reason: sanitizeReason(
            decision.reason,
            "No meaningful Discord evidence has been collected yet."
        ),
    };
}

export function fallbackEvidenceDecision(
    state: Pick<
        GraphState,
        "question" | "toolHistory" | "evidence" | "actorId" | "candidateCapabilities"
    >
): EvidenceDecision {
    return normalizeEvidenceDecision(state, {
        sufficient: false,
        confidence: "insufficient",
        reason: "No meaningful Discord evidence has been collected yet.",
    });
}

export async function judgeEvidence(
    state: Pick<
        GraphState,
        "question" | "toolHistory" | "evidence" | "actorId" | "candidateCapabilities"
    >
): Promise<EvidenceDecision> {
    const fallback = fallbackEvidenceDecision(state);
    try {
        const raw = await ModelGateway.generateJson<EvidenceDecision>(
            [
                { role: "system", content: PromptRegistry.load("system/base") },
                {
                    role: "user",
                    content: PromptRegistry.render("runtime/judge_evidence", {
                        question: state.question,
                        evidence: summarizeEvidence(state),
                    }),
                },
            ],
            fallback,
            {
                traceContext: {
                    traceLabel: "runtime_judge_evidence",
                    questionPreview: state.question,
                },
            }
        );
        return normalizeEvidenceDecision(state, raw);
    } catch {
        return fallback;
    }
}

function normalizeStepDecision(
    state: Pick<
        GraphState,
        | "question"
        | "candidateCapabilities"
        | "toolHistory"
        | "actorId"
        | "replyContext"
        | "activeMemberTarget"
        | "activeChannelTarget"
        | "activeResolvedChannelIds"
        | "activeRetrievalSession"
        | "turnIntent"
    >,
    step: StepDecision
): StepDecision {
    const ambiguousMemberCandidate = extractAmbiguousMemberCandidate(state.toolHistory);
    const nextCapability =
        step.nextCapability && VALID_TOOL_NAMES.has(step.nextCapability)
            ? step.nextCapability
            : null;

    if (!nextCapability) {
        return {
            nextCapability: null,
            arguments: {},
            reason: "No valid next capability was selected.",
            learnedExpectation: "Stop and answer with the current evidence.",
        };
    }

    const argumentsObject = sanitizeArguments(step.arguments);
    const resolvedMember = extractLatestResolvedMember(state.toolHistory);
    const latestResolvedChannelTarget = extractLatestResolvedChannelTarget(state.toolHistory);
    const resolvedChannelIds =
        state.activeResolvedChannelIds.length > 0
            ? state.activeResolvedChannelIds
            : latestResolvedChannelTarget?.resolvedIds || extractLatestResolvedChannelIds(state.toolHistory);
    const structuralMember = extractStructuralMemberReference(state.question, state.replyContext);
    const structuralChannel = extractStructuralChannelReference(state.question, state.replyContext);
    const activeMember = state.activeMemberTarget;
    const activeChannelTarget = state.activeChannelTarget || latestResolvedChannelTarget;
    const activeRetrievalSession = state.activeRetrievalSession;
    const explicitBeforeTimestamp = parseFlexibleTimestamp(argumentsObject.beforeTimestamp);
    const explicitAfterTimestamp = parseFlexibleTimestamp(argumentsObject.afterTimestamp);
    const timeBounds = {
        beforeTimestamp: state.turnIntent?.beforeTimestamp ?? undefined,
        afterTimestamp: state.turnIntent?.afterTimestamp ?? undefined,
    };
    const resetRetrievalSession =
        Boolean(
            structuralChannel &&
                activeChannelTarget?.query &&
                normalize(structuralChannel) !== normalize(activeChannelTarget.query)
        ) ||
        Boolean(
            resolvedChannelIds.length &&
                activeRetrievalSession?.channelIds.length &&
                JSON.stringify(resolvedChannelIds) !== JSON.stringify(activeRetrievalSession.channelIds)
        );
    const shouldContinueSession =
        !resetRetrievalSession && state.turnIntent?.continuation === true;

    if (nextCapability === "resolve_member_identity") {
        return {
            nextCapability,
            arguments: {
                query:
                    typeof argumentsObject.query === "string" && argumentsObject.query.trim()
                        ? argumentsObject.query.trim()
                        : structuralMember || activeMember?.resolvedId || state.actorId,
            },
            reason: sanitizeReason(
                step.reason,
                "Resolve the relevant member or bot before answering."
            ),
            learnedExpectation: sanitizeReason(
                step.learnedExpectation,
                "Return the best current-guild identity match or historical author fallback."
            ),
        };
    }

    if (nextCapability === "resolve_channel_targets") {
        return {
            nextCapability,
            arguments: {
                targetText:
                    typeof argumentsObject.targetText === "string" && argumentsObject.targetText.trim()
                        ? argumentsObject.targetText.trim()
                        : structuralChannel || activeChannelTarget?.query || state.question,
            },
            reason: sanitizeReason(
                step.reason,
                "Resolve the referenced channel or category before answering."
            ),
            learnedExpectation: sanitizeReason(
                step.learnedExpectation,
                "Return exact message-channel ids for the current guild target."
            ),
        };
    }

    if (nextCapability === "retrieve_messages") {
        const forensicScopedLookup = Boolean(
            timeBounds.beforeTimestamp != null ||
                timeBounds.afterTimestamp != null ||
                structuralMember
        );

        const resolvedBeforeTimestamp =
            timeBounds.beforeTimestamp ??
            explicitBeforeTimestamp ??
            activeRetrievalSession?.beforeTimestamp;
        const resolvedAfterTimestamp =
            timeBounds.afterTimestamp ??
            explicitAfterTimestamp ??
            activeRetrievalSession?.afterTimestamp;

        return {
            nextCapability,
            arguments: {
                query:
                    typeof argumentsObject.query === "string" && argumentsObject.query.trim()
                        ? argumentsObject.query.trim()
                        : state.question,
                mode:
                    typeof argumentsObject.mode === "string" &&
                    ["history", "semantic", "mixed"].includes(argumentsObject.mode)
                        ? argumentsObject.mode
                        : state.turnIntent?.retrievalMode ?? activeRetrievalSession?.mode ?? "history",
                limit:
                    typeof argumentsObject.limit === "number" ? argumentsObject.limit : 8,
                ...(resolvedMember?.resolvedId || activeMember?.resolvedId
                    ? { authorId: resolvedMember?.resolvedId || activeMember?.resolvedId }
                    : {}),
                ...(resolvedChannelIds.length
                    ? { channelIds: resolvedChannelIds }
                    : {}),
                ...(resolvedBeforeTimestamp != null
                    ? {
                          beforeTimestamp: resolvedBeforeTimestamp,
                      }
                    : {}),
                ...(resolvedAfterTimestamp != null
                    ? {
                          afterTimestamp: resolvedAfterTimestamp,
                      }
                    : {}),
                ...(shouldContinueSession &&
                activeRetrievalSession?.historyCursorByChannel &&
                Object.keys(activeRetrievalSession.historyCursorByChannel).length
                    ? {
                          cursor: ({
                              history: activeRetrievalSession.historyCursorByChannel,
                              ...(activeRetrievalSession.semanticCursor
                                  ? { semantic: activeRetrievalSession.semanticCursor }
                                  : {}),
                          } as unknown as ToolArgumentValue),
                      }
                    : {}),
                    ...(shouldContinueSession && activeRetrievalSession?.seenMessageIds?.length
                    ? {
                          excludedMessageIds: activeRetrievalSession.seenMessageIds,
                      }
                    : {}),
            },
            reason: sanitizeReason(
                step.reason,
                "Read scoped Discord history first, then supplement with semantic matches if needed."
            ),
            learnedExpectation: sanitizeReason(
                step.learnedExpectation,
                "Return ordered scoped history, semantic matches, and a continuation cursor."
            ),
        };
    }

    if (nextCapability === "get_member_profile") {
        const profiled = ambiguousMemberCandidate
            ? extractProfiledMemberIdentifiers(state.toolHistory)
            : null;
        const nextUnprofiled = ambiguousMemberCandidate?.identifiers.find(
            (id) => profiled && !profiled.has(normalize(id)) && !profiled.has(id)
        );
        const modelProvided =
            typeof argumentsObject.nameOrId === "string" && argumentsObject.nameOrId.trim()
                ? argumentsObject.nameOrId.trim()
                : null;
        // When there are unprofiled ambiguous members, prefer the specific identifier
        // over the model's argument — the model often passes the shared display name
        // which resolves to the already-profiled member again.
        const nameOrId =
            nextUnprofiled ||
            modelProvided ||
            ambiguousMemberCandidate?.displayName ||
            resolvedMember?.resolvedId ||
            activeMember?.resolvedId ||
            structuralMember ||
            state.question;
        return {
            nextCapability,
            arguments: { nameOrId },
            reason: sanitizeReason(
                step.reason,
                "Fetch member profile details after identity resolution."
            ),
            learnedExpectation: sanitizeReason(
                step.learnedExpectation,
                "Return the current guild member profile when available."
            ),
        };
    }

    if (nextCapability === "list_guild_structure") {
        return {
            nextCapability,
            arguments: {
                ...(typeof argumentsObject.targetText === "string" && argumentsObject.targetText.trim()
                    ? { targetText: argumentsObject.targetText.trim() }
                    : activeChannelTarget?.query
                      ? { targetText: activeChannelTarget.query }
                      : structuralChannel
                        ? { targetText: structuralChannel }
                        : { targetText: state.question }),
            },
            reason: sanitizeReason(
                step.reason,
                "Inspect the current guild structure around the resolved or likely category/channel target."
            ),
            learnedExpectation: sanitizeReason(
                step.learnedExpectation,
                "Return matched categories/channels plus visible child-channel structure."
            ),
        };
    }

    if (nextCapability === "list_members") {
        return {
            nextCapability,
            arguments: {
                filters:
                    typeof argumentsObject.filters === "string"
                        ? argumentsObject.filters
                        : undefined,
                limit:
                    typeof argumentsObject.limit === "number" ? argumentsObject.limit : 20,
                offset:
                    typeof argumentsObject.offset === "number" ? argumentsObject.offset : 0,
            },
            reason: sanitizeReason(
                step.reason,
                "List guild members when broader identity context may help."
            ),
            learnedExpectation: sanitizeReason(
                step.learnedExpectation,
                "Return the relevant current-guild members."
            ),
        };
    }

    return {
        nextCapability,
        arguments: argumentsObject,
        reason: sanitizeReason(step.reason, "Use the selected capability."),
        learnedExpectation: sanitizeReason(
            step.learnedExpectation,
            "Use the capability output to improve the answer."
        ),
    };
}

export function fallbackStepDecision(
    state: Pick<
        GraphState,
        | "question"
        | "candidateCapabilities"
        | "toolHistory"
        | "actorId"
        | "replyContext"
        | "activeMemberTarget"
        | "activeChannelTarget"
        | "activeResolvedChannelIds"
        | "activeRetrievalSession"
        | "turnIntent"
    >
): StepDecision {
    const resolvedMember = extractLatestResolvedMember(state.toolHistory);
    const latestResolvedChannelTarget = extractLatestResolvedChannelTarget(state.toolHistory);
    const resolvedChannelIds =
        state.activeResolvedChannelIds.length > 0
            ? state.activeResolvedChannelIds
            : latestResolvedChannelTarget?.resolvedIds || extractLatestResolvedChannelIds(state.toolHistory);
    const structuralMember = extractStructuralMemberReference(state.question, state.replyContext);
    const structuralChannel = extractStructuralChannelReference(state.question, state.replyContext);
    const activeChannelTarget = state.activeChannelTarget || latestResolvedChannelTarget;
    const activeRetrievalSession = state.activeRetrievalSession;
    const changedScope =
        Boolean(
            structuralChannel &&
                activeChannelTarget?.query &&
                normalize(structuralChannel) !== normalize(activeChannelTarget.query)
        );
    const forensicScopedLookup =
        resolvedChannelIds.length > 0 &&
        Boolean(
            state.turnIntent?.beforeTimestamp != null ||
                state.turnIntent?.afterTimestamp != null ||
                structuralMember
        );
    const ambiguousMemberCandidate = extractAmbiguousMemberCandidate(state.toolHistory);

    if (
        activeRetrievalSession?.continuationAvailable &&
        !changedScope &&
        state.turnIntent?.continuation === true
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "retrieve_messages",
            arguments: {
                query: state.question,
            },
            reason: "Continue the active scoped history read without restarting from the beginning.",
            learnedExpectation: "Return the next non-duplicate page from the active retrieval session.",
        });
    }

    if (
        changedScope &&
        structuralChannel
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "resolve_channel_targets",
            arguments: { targetText: structuralChannel },
            reason: "Resolve the exact channel or category reference before broader retrieval.",
            learnedExpectation: "Return exact message-channel ids for the current guild target.",
        });
    }

    if (
        structuralMember &&
        !resolvedMember?.resolvedId &&
        !hasToolRun(state.toolHistory, "resolve_member_identity")
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "resolve_member_identity",
            arguments: { query: structuralMember },
            reason: "Resolve the exact member reference before broader retrieval.",
            learnedExpectation: "Return the best current-guild identity match or historical fallback.",
        });
    }

    if (
        structuralChannel &&
        !resolvedChannelIds.length &&
        !hasToolRun(state.toolHistory, "resolve_channel_targets")
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "resolve_channel_targets",
            arguments: { targetText: structuralChannel },
            reason: "Resolve the exact channel or category reference before broader retrieval.",
            learnedExpectation: "Return exact message-channel ids for the current guild target.",
        });
    }

    if (ambiguousMemberCandidate) {
        const profiled = extractProfiledMemberIdentifiers(state.toolHistory);
        const unprofiled = ambiguousMemberCandidate.identifiers.find(
            (id) => !profiled.has(normalize(id)) && !profiled.has(id)
        );
        if (unprofiled) {
            return normalizeStepDecision(state, {
                nextCapability: "get_member_profile",
                arguments: { nameOrId: unprofiled },
                reason:
                    "Multiple current-guild members share the same visible name; fetch profile details for each to disambiguate safely.",
                learnedExpectation:
                    "Return a distinguishing profile for the ambiguous member so profiles can be compared.",
            });
        }
    }

    if (
        !activeChannelTarget &&
        !resolvedChannelIds.length &&
        !hasToolRun(state.toolHistory, "resolve_channel_targets")
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "resolve_channel_targets",
            arguments: {
                targetText: structuralChannel || state.question,
            },
            reason: "Resolve the most likely channel or category target before broader discovery.",
            learnedExpectation: "Return exact message-channel ids for the likely current-guild target.",
        });
    }

    if (
        resolvedChannelIds.length &&
        !hasToolRun(state.toolHistory, "retrieve_messages")
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "retrieve_messages",
            arguments: {
                query: state.question,
                channelIds: resolvedChannelIds,
            },
            reason: "Use the resolved channel scope to retrieve Discord messages before answering.",
            learnedExpectation: "Return scoped message evidence from the resolved category or channel area.",
        });
    }

    if (
        !forensicScopedLookup &&
        (activeChannelTarget || resolvedChannelIds.length) &&
        !hasToolRun(state.toolHistory, "list_guild_structure")
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "list_guild_structure",
            arguments: {
                targetText: activeChannelTarget?.query || structuralChannel || state.question,
            },
            reason: "Inspect the matched category or channel structure before summarizing it.",
            learnedExpectation: "Return the matched structure plus visible child channels.",
        });
    }

    const nextCapability =
        GENERIC_STEP_ORDER.find(
            (tool) => !hasToolRun(state.toolHistory, tool)
        ) || null;

    if (!nextCapability) {
        return {
            nextCapability: null,
            arguments: {},
            reason: "No unused capability remains likely to improve the answer.",
            learnedExpectation: "Stop and answer with the current evidence.",
        };
    }

    return normalizeStepDecision(state, {
        nextCapability,
        arguments: {},
        reason: "Use the next generic current-guild capability in the recovery ladder.",
        learnedExpectation: "Use the next capability to improve grounding before answering.",
    });
}

export async function planNextStep(
    state: Pick<
        GraphState,
        | "question"
        | "goal"
        | "successCriteria"
        | "confidence"
        | "toolHistory"
        | "candidateCapabilities"
        | "actorId"
        | "replyContext"
        | "activeMemberTarget"
        | "activeChannelTarget"
        | "activeResolvedChannelIds"
        | "activeRetrievalSession"
        | "turnIntent"
    >
): Promise<StepDecision> {
    const fallback = fallbackStepDecision(state);
    try {
        const raw = await ModelGateway.generateJson<StepDecision>(
            [
                { role: "system", content: PromptRegistry.load("system/base") },
                {
                    role: "user",
                    content: PromptRegistry.render("runtime/select_next_step", {
                        question: state.question,
                        goal: state.goal,
                        success_criteria: state.successCriteria,
                        confidence: state.confidence,
                        tool_history: summarizeToolHistory(state.toolHistory),
                        active_member_target: state.activeMemberTarget
                            ? `${state.activeMemberTarget.displayName} (@${state.activeMemberTarget.username})`
                            : "none",
                        active_channel_target: state.activeChannelTarget
                            ? `${state.activeChannelTarget.query} -> ${state.activeResolvedChannelIds.join(", ") || "no resolved channel ids"}`
                            : state.activeResolvedChannelIds.length
                              ? state.activeResolvedChannelIds.join(", ")
                              : "none",
                        active_retrieval_session: state.activeRetrievalSession
                            ? `${state.activeRetrievalSession.mode} :: ${state.activeRetrievalSession.channelIds.join(", ")} :: continuation=${state.activeRetrievalSession.continuationAvailable ? "yes" : "no"}`
                            : "none",
                        capability_registry: CapabilityRegistry.describeForPrompt(),
                    }),
                },
            ],
            fallback,
            {
                traceContext: {
                    traceLabel: "runtime_select_next_step",
                    questionPreview: state.question,
                },
            }
        );

        const normalized = normalizeStepDecision(state, raw);
        return normalized.nextCapability ? normalized : fallback;
    } catch {
        return fallback;
    }
}

export function answerConfidenceForInsufficient(
    state: Pick<GraphState, "evidence">
): GroundedAnswerMode {
    const summary = countEvidence(state);
    if (summary.strongMessageEvidenceCount > 0) {
        return "best_effort";
    }
    if (summary.messageEvidenceCount >= 2) {
        return "best_effort";
    }
    return "insufficient";
}

export function formatChannelContext(context: ChannelContextMessage[]): string {
    if (!context.length) {
        return "No recent channel messages available.";
    }
    return context.map((msg) => `${msg.authorName}: ${msg.content}`).join("\n");
}

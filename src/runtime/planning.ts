import { ModelGateway } from "@/ai/ModelGateway";
import { CapabilityRegistry } from "@/discord/capabilities/CapabilityRegistry";
import { PromptRegistry } from "@/runtime/PromptRegistry";
import type {
    ChannelContextMessage,
    EvidenceDecision,
    GraphState,
    PlanDecision,
    RuntimeMode,
    StepDecision,
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

function normalize(text: string): string {
    return text
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .trim();
}

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
    return (
        extractMemberMentionId(question) ||
        extractBareSnowflake(question) ||
        null
    );
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

function sanitizeArguments(
    value: unknown
): Record<string, string | number | undefined> {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return {};
    }

    const entries = Object.entries(value as Record<string, unknown>).filter(
        ([, raw]) =>
            raw === undefined || typeof raw === "string" || typeof raw === "number"
    );

    return Object.fromEntries(entries) as Record<string, string | number | undefined>;
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
                  return `${item.tool}: ${meta ? `[${meta}] ` : ""}${item.content}`;
              })
              .join("\n")
        : "No Discord evidence collected.";
}

export function countEvidence(state: Pick<GraphState, "evidence">) {
    return state.evidence.reduce(
        (acc, item) => {
            if (item.evidenceRole === "message_evidence") {
                acc.messageEvidenceCount += 1;
            }
            if (
                item.evidenceRole === "live_evidence" ||
                item.evidenceRole === "discovery_only"
            ) {
                acc.liveEvidenceCount += 1;
            }
            return acc;
        },
        { messageEvidenceCount: 0, liveEvidenceCount: 0 }
    );
}

export function guessPlan(input: TurnInput): PlanDecision {
    const mode = classifyFallbackMode(input);
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
    });
}

export async function planWithModel(
    input: TurnInput,
    channelContext: ChannelContextMessage[] = [],
    recentTurns: Pick<GraphState, "recentTurns">["recentTurns"] = [],
    activeTargets?: Pick<
        GraphState,
        "activeMemberTarget" | "activeChannelTarget" | "activeResolvedChannelIds"
    >
): Promise<PlanDecision> {
    const fallback = guessPlan(input);
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
        return normalizePlanDecision(input, raw);
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
    >,
    step: StepDecision
): StepDecision {
    const nextCapability =
        step.nextCapability &&
        VALID_TOOL_NAMES.has(step.nextCapability) &&
        state.candidateCapabilities.includes(step.nextCapability)
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
        if (
            structuralChannel &&
            !resolvedChannelIds.length &&
            state.candidateCapabilities.includes("resolve_channel_targets") &&
            !hasToolRun(state.toolHistory, "resolve_channel_targets")
        ) {
            return normalizeStepDecision(state, {
                nextCapability: "resolve_channel_targets",
                arguments: {
                    targetText: structuralChannel,
                },
                reason: "Resolve the referenced channel or category before unscoped retrieval.",
                learnedExpectation: "Return exact message-channel ids for the named current-guild target.",
            });
        }

        if (
            resolvedChannelIds.length &&
            state.candidateCapabilities.includes("list_guild_structure") &&
            !hasToolRun(state.toolHistory, "list_guild_structure")
        ) {
            return normalizeStepDecision(state, {
                nextCapability: "list_guild_structure",
                arguments: {
                    targetText: activeChannelTarget?.query || structuralChannel || state.question,
                },
                reason: "Inspect the resolved category or channel structure before scoped retrieval.",
                learnedExpectation: "Confirm the visible child channels before reading scoped messages.",
            });
        }

        return {
            nextCapability,
            arguments: {
                query:
                    typeof argumentsObject.query === "string" && argumentsObject.query.trim()
                        ? argumentsObject.query.trim()
                        : state.question,
                limit:
                    typeof argumentsObject.limit === "number" ? argumentsObject.limit : 8,
                ...(resolvedMember?.resolvedId || activeMember?.resolvedId
                    ? { authorId: resolvedMember?.resolvedId || activeMember?.resolvedId }
                    : {}),
                ...(resolvedChannelIds.length
                    ? { channelIds: resolvedChannelIds.join(",") }
                    : {}),
            },
            reason: sanitizeReason(
                step.reason,
                "Search Discord messages using the current cache-first retrieval pipeline."
            ),
            learnedExpectation: sanitizeReason(
                step.learnedExpectation,
                "Search the local cache first, then escalate live if needed."
            ),
        };
    }

    if (
        nextCapability === "list_guild_structure" &&
        !activeChannelTarget &&
        !resolvedChannelIds.length &&
        state.candidateCapabilities.includes("resolve_channel_targets") &&
        !hasToolRun(state.toolHistory, "resolve_channel_targets")
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "resolve_channel_targets",
            arguments: {
                targetText:
                    typeof argumentsObject.targetText === "string" && argumentsObject.targetText.trim()
                        ? argumentsObject.targetText.trim()
                        : structuralChannel || state.question,
            },
            reason: "Resolve the likely category or channel target before inspecting guild structure.",
            learnedExpectation: "Return the exact or best-matched guild target before structure inspection.",
        });
    }

    if (nextCapability === "get_member_profile") {
        return {
            nextCapability,
            arguments: {
                nameOrId:
                    typeof argumentsObject.nameOrId === "string" && argumentsObject.nameOrId.trim()
                        ? argumentsObject.nameOrId.trim()
                        : resolvedMember?.resolvedId ||
                          activeMember?.resolvedId ||
                          structuralMember ||
                          state.question,
            },
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
                        : state.question,
                limit:
                    typeof argumentsObject.limit === "number" ? argumentsObject.limit : 10,
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

    if (
        structuralMember &&
        !resolvedMember?.resolvedId &&
        state.candidateCapabilities.includes("resolve_member_identity") &&
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
        state.candidateCapabilities.includes("resolve_channel_targets") &&
        !hasToolRun(state.toolHistory, "resolve_channel_targets")
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "resolve_channel_targets",
            arguments: { targetText: structuralChannel },
            reason: "Resolve the exact channel or category reference before broader retrieval.",
            learnedExpectation: "Return exact message-channel ids for the current guild target.",
        });
    }

    if (
        !activeChannelTarget &&
        !resolvedChannelIds.length &&
        state.candidateCapabilities.includes("resolve_channel_targets") &&
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
        (activeChannelTarget || resolvedChannelIds.length) &&
        state.candidateCapabilities.includes("list_guild_structure") &&
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

    if (
        resolvedChannelIds.length &&
        state.candidateCapabilities.includes("retrieve_messages") &&
        !hasToolRun(state.toolHistory, "retrieve_messages")
    ) {
        return normalizeStepDecision(state, {
            nextCapability: "retrieve_messages",
            arguments: {
                query: state.question,
                channelIds: resolvedChannelIds.join(","),
            },
            reason: "Use the resolved channel scope to retrieve Discord messages before answering.",
            learnedExpectation: "Return scoped message evidence from the resolved category or channel area.",
        });
    }

    const nextCapability =
        GENERIC_STEP_ORDER.find(
            (tool) =>
                state.candidateCapabilities.includes(tool) && !hasToolRun(state.toolHistory, tool)
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
    if (summary.messageEvidenceCount > 0 || summary.liveEvidenceCount > 0) {
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

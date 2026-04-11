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

function extractStructuralMemberReference(
    question: string,
    replyContext?: Pick<TurnInput, "replyContext">["replyContext"] | null
): string | null {
    return (
        extractMemberMentionId(question) ||
        replyContext?.authorId ||
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
        (replyContext?.content ? extractChannelMentionId(replyContext.content) : null) ||
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

function normalizePlanDecision(input: TurnInput, plan: PlanDecision): PlanDecision {
    let mode: RuntimeMode = plan.mode === "research" ? "research" : "conversation";
    let candidateCapabilities = sanitizeCandidateCapabilities(plan.candidateCapabilities);

    if (!input.guild) {
        mode = "conversation";
        candidateCapabilities = [];
    } else if (mode === "research") {
        candidateCapabilities = ensureResearchCapabilities(candidateCapabilities);
    } else {
        candidateCapabilities = [];
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
            if (item.evidenceRole === "live_evidence") {
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
    channelContext: ChannelContextMessage[] = []
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
                        recent_turns: input.replyContext ? "reply context available" : "",
                        channel_context: formatChannelContext(channelContext),
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
    state: Pick<GraphState, "question" | "toolHistory" | "evidence" | "actorId">,
    decision: EvidenceDecision
): EvidenceDecision {
    const retrieval = extractLatestRetrieval(state.toolHistory);
    const memberProfileId = extractLatestMemberProfileId(state.toolHistory);
    const counts = countEvidence(state);

    if (memberProfileId === state.actorId && counts.messageEvidenceCount === 0) {
        return {
            sufficient: true,
            confidence: "confident",
            reason: "The requesting member was resolved directly.",
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
    state: Pick<GraphState, "question" | "toolHistory" | "evidence" | "actorId">
): EvidenceDecision {
    return normalizeEvidenceDecision(state, {
        sufficient: false,
        confidence: "insufficient",
        reason: "No meaningful Discord evidence has been collected yet.",
    });
}

export async function judgeEvidence(
    state: Pick<GraphState, "question" | "toolHistory" | "evidence" | "actorId">
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
        "question" | "candidateCapabilities" | "toolHistory" | "actorId" | "replyContext"
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
    const resolvedChannelIds = extractLatestResolvedChannelIds(state.toolHistory);
    const structuralMember = extractStructuralMemberReference(state.question, state.replyContext);
    const structuralChannel = extractStructuralChannelReference(state.question, state.replyContext);

    if (nextCapability === "resolve_member_identity") {
        return {
            nextCapability,
            arguments: {
                query:
                    typeof argumentsObject.query === "string" && argumentsObject.query.trim()
                        ? argumentsObject.query.trim()
                        : structuralMember || state.actorId,
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
                        : structuralChannel || state.question,
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
        return {
            nextCapability,
            arguments: {
                query:
                    typeof argumentsObject.query === "string" && argumentsObject.query.trim()
                        ? argumentsObject.query.trim()
                        : state.question,
                limit:
                    typeof argumentsObject.limit === "number" ? argumentsObject.limit : 8,
                ...(resolvedMember?.resolvedId ? { authorId: resolvedMember.resolvedId } : {}),
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

    if (nextCapability === "get_member_profile") {
        return {
            nextCapability,
            arguments: {
                nameOrId:
                    typeof argumentsObject.nameOrId === "string" && argumentsObject.nameOrId.trim()
                        ? argumentsObject.nameOrId.trim()
                        : resolvedMember?.resolvedId || structuralMember || state.question,
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
        "question" | "candidateCapabilities" | "toolHistory" | "actorId" | "replyContext"
    >
): StepDecision {
    const resolvedMember = extractLatestResolvedMember(state.toolHistory);
    const resolvedChannelIds = extractLatestResolvedChannelIds(state.toolHistory);
    const structuralMember = extractStructuralMemberReference(state.question, state.replyContext);
    const structuralChannel = extractStructuralChannelReference(state.question, state.replyContext);

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
        "question" | "goal" | "successCriteria" | "confidence" | "toolHistory" | "candidateCapabilities" | "actorId" | "replyContext"
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

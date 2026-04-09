import type { Guild } from "discord.js";
import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";
import { extractBestChunk } from "@/discord/tools/runtime/resultReaders";
import {
    assessGrounding,
    collectMemberEvidence,
    memberEvidenceNeedsMore,
} from "@/agent/orchestration/grounding";
import {
    extractLikelyPersonName,
    extractMentionedChannelIds,
    extractMentionedUserIds,
    extractRequestedOrdinal,
    extractTopicHint,
    isIdentityQuestion,
    isPersonMessageQuestion,
    isReferentialFollowUp,
    normalizeQuestion,
    requestsAllMembers,
} from "@/agent/orchestration/questionAnalysis";
import type { ConversationResolutionContext } from "@/shared/appTypes";
import type {
    ChannelCandidate,
    DiscordToolResult,
    GroundedAnswerMode,
    QuestionIntent,
    RetrievalControllerDecision,
    RouteIntent,
} from "@/shared/appTypes";
import type { SearchContext } from "@/agent/orchestration/types";

type RetrievalControllerInput = {
    question: string;
    guild: Guild | null;
    currentChannelId?: string | null;
    toolRuns: DiscordToolResult[];
    priorContext?: ConversationResolutionContext | null;
    context: SearchContext;
    forcedFinal?: boolean;
};

type RetrievalControllerModelDecision = {
    questionIntent?: QuestionIntent;
    nextAction?: RetrievalControllerDecision["nextAction"];
    targetText?: string;
    authorId?: string;
    authorQuery?: string;
    topicText?: string;
    channelHintText?: string;
    channelIds?: string[];
    searchQuery?: string;
    needsMessageEvidence?: boolean;
    answerConfidence?: GroundedAnswerMode;
    confidence?: number;
    reason?: string;
};

export async function decideRetrievalAction(
    input: RetrievalControllerInput
): Promise<RetrievalControllerDecision> {
    const overrides = getDeterministicOverrides(input.question);
    const readableChannels = DiscordLiveService.listReadableGuildChannels(input.guild);
    const grounding = assessGrounding(input.question, input.toolRuns);
    const fallback = buildFallbackDecision(input, overrides, grounding.summary);
    let decision = toModelFallback(fallback);
    let usedModel = false;
    try {
        decision = await ModelGateway.generateJson<RetrievalControllerModelDecision>(
            [
                { role: "system", content: "Return strict JSON only." },
                {
                    role: "user",
                    content: PromptRegistry.render("tasks/retrieval_controller", {
                        question: input.question,
                        current_channel_id: input.currentChannelId || "",
                        forced_final: input.forcedFinal ? "true" : "false",
                        explicit_user_ids: overrides.authorIds.length
                            ? overrides.authorIds.join(", ")
                            : "none",
                        explicit_channel_ids: overrides.channelIds.length
                            ? overrides.channelIds.join(", ")
                            : "none",
                        readable_channels: readableChannels.length
                            ? readableChannels
                                  .map((channel) => `${channel.name} (${channel.id})`)
                                  .join(", ")
                            : "none",
                        extracted_topic_hint: extractTopicHint(input.question) || "none",
                        extracted_person_hint:
                            extractLikelyPersonName(input.question) || "none",
                        referential_follow_up: isReferentialFollowUp(input.question)
                            ? "true"
                            : "false",
                        prior_context: input.priorContext
                            ? JSON.stringify(input.priorContext, null, 2)
                            : "none",
                        grounding_summary: JSON.stringify(grounding.summary, null, 2),
                        tool_history: summarizeToolHistory(input.toolRuns),
                        crawled_channel_ids:
                            [...input.context.crawledChannelIds].join(", ") || "none",
                        seeded_from_context: input.context.seededFromContext ? "true" : "false",
                        initial_tool_runs_count: input.context.initialToolRuns.length,
                    }),
                },
            ],
            toModelFallback(fallback),
            {
                traceContext: {
                    traceLabel: "discord_retrieval_controller",
                    questionPreview: input.question,
                },
            }
        );
        usedModel = true;
    } catch {
        decision = toModelFallback(fallback);
    }

    return normalizeControllerDecision(
        decision,
        fallback,
        input,
        readableChannels,
        usedModel
    );
}

function toModelFallback(
    fallback: RetrievalControllerDecision
): RetrievalControllerModelDecision {
    return {
        questionIntent: fallback.questionIntent,
        nextAction: fallback.nextAction,
        targetText: fallback.targetText ?? undefined,
        authorId: fallback.authorId,
        authorQuery: fallback.authorQuery,
        topicText: fallback.topicText ?? undefined,
        channelHintText: fallback.channelHintText ?? undefined,
        channelIds: fallback.channelIds,
        searchQuery: fallback.searchQuery ?? undefined,
        needsMessageEvidence: fallback.needsMessageEvidence,
        answerConfidence: fallback.answerConfidence,
        confidence: fallback.confidence,
        reason: fallback.reason,
    };
}

export function mapQuestionIntentToRouteIntent(intent: QuestionIntent): RouteIntent {
    switch (intent) {
        case "person_identity":
        case "person_messages":
            return "person_target";
        case "member_list_or_ordinal":
            return "member_lookup";
        case "server_context":
            return "server_context";
        case "channel_or_topic_search":
            return "channel_target";
        case "broad_search":
        default:
            return "broad_search";
    }
}

function getDeterministicOverrides(question: string) {
    return {
        authorIds: extractMentionedUserIds(question),
        channelIds: extractMentionedChannelIds(question),
    };
}

function buildFallbackDecision(
    input: RetrievalControllerInput,
    overrides: ReturnType<typeof getDeterministicOverrides>,
    groundingSummary: { messageEvidenceCount: number; liveEvidenceCount: number; sufficient: boolean }
): RetrievalControllerDecision {
    const normalized = normalizeQuestion(input.question);
    const latestSearchRun = [...input.toolRuns]
        .reverse()
        .find((run) => run.tool === "search_messages");
    const latestChannelListRun = [...input.toolRuns]
        .reverse()
        .find((run) => run.tool === "list_relevant_channels");
    const latestCrawlRun = [...input.toolRuns]
        .reverse()
        .find((run) => run.tool === "crawl_channel_messages");
    const latestCrawlPreviewCount =
        latestCrawlRun && latestCrawlRun.data
            ? (((latestCrawlRun.data as any).previewMessages as any[] | undefined)?.length ?? 0)
            : 0;
    const latestProfile = [...input.toolRuns]
        .reverse()
        .find((run) => run.tool === "get_member_profile");
    const profile = latestProfile
        ? DiscordToolService.getMemberProfileResult(latestProfile)
        : null;
    const memberEvidence = collectMemberEvidence(input.toolRuns);
    const ordinal = extractRequestedOrdinal(input.question);
    const topicText = extractTopicHint(input.question);
    const likelyTarget =
        extractLikelyPersonName(input.question) ??
        input.priorContext?.authorQuery ??
        input.priorContext?.targetText ??
        null;
    const needsMessageEvidence = isPersonMessageQuestion(input.question);
    const questionIntent: QuestionIntent = overrides.authorIds.length
        ? needsMessageEvidence
            ? "person_messages"
            : "person_identity"
        : overrides.channelIds.length
          ? "channel_or_topic_search"
        : ordinal !== null || requestsAllMembers(input.question)
          ? "member_list_or_ordinal"
          : normalized.includes("canal") || normalized.includes("channel")
            ? "channel_or_topic_search"
          : normalized.includes("servidor") || normalized.includes("server")
            ? "server_context"
            : needsMessageEvidence
              ? "person_messages"
              : isIdentityQuestion(input.question)
                ? "person_identity"
                : topicText
                  ? "channel_or_topic_search"
                  : "broad_search";
    const routeIntent = mapQuestionIntentToRouteIntent(questionIntent);
    const targetChannelIds = overrides.channelIds.length
        ? overrides.channelIds
        : input.priorContext?.channelIds ?? [];
    const bestChunk = extractBestChunk(input.toolRuns);
    const hasStrongMessageEvidence =
        groundingSummary.messageEvidenceCount > 0 &&
        Boolean(bestChunk && bestChunk.totalScore >= 0.45);
    const hasAnyEvidence =
        groundingSummary.messageEvidenceCount > 0 || groundingSummary.liveEvidenceCount > 0;

    if (input.forcedFinal) {
        if (questionIntent === "server_context" && groundingSummary.liveEvidenceCount > 0) {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "answer",
                targetText: null,
                needsMessageEvidence: false,
                answerConfidence: "confident",
                confidence: 0.5,
                reason: "Forced final pass with usable server context.",
            };
        }

        if (
            (questionIntent === "person_identity" &&
                (profile || groundingSummary.liveEvidenceCount > 0)) ||
            groundingSummary.messageEvidenceCount > 0 ||
            groundingSummary.liveEvidenceCount > 0
        ) {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "best_effort_answer",
                targetText: likelyTarget,
                authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                authorQuery: likelyTarget ?? undefined,
                topicText,
                channelHintText: topicText,
                channelIds: targetChannelIds.length ? targetChannelIds : undefined,
                searchQuery: topicText ?? input.question,
                needsMessageEvidence,
                answerConfidence:
                    questionIntent === "person_identity" && (profile || groundingSummary.liveEvidenceCount > 0)
                        ? "confident"
                        : "best_effort",
                confidence: 0.35,
                reason: "Forced final pass with partial evidence.",
            };
        }

        return {
            source: "deterministic",
            questionIntent,
            routeIntent,
            nextAction: "answer",
            targetText: likelyTarget,
            authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
            authorQuery: likelyTarget ?? undefined,
            topicText,
            channelHintText: topicText,
            channelIds: targetChannelIds.length ? targetChannelIds : undefined,
            searchQuery: topicText ?? input.question,
            needsMessageEvidence,
            answerConfidence: "insufficient",
            confidence: 0.3,
            reason: "Forced final pass without useful evidence.",
        };
    }

    if (questionIntent === "server_context" && groundingSummary.liveEvidenceCount > 0) {
        return {
            source: "deterministic",
            questionIntent,
            routeIntent,
            nextAction: "answer",
            targetText: null,
            needsMessageEvidence: false,
            answerConfidence: "confident",
            confidence: 0.7,
            reason: "Live guild context already exists.",
        };
    }

    if (
        questionIntent === "member_list_or_ordinal" &&
        memberEvidence &&
        memberEvidenceNeedsMore(input.question, memberEvidence, ordinal)
    ) {
        return {
            source: "deterministic",
            questionIntent,
            routeIntent,
            nextAction: "list_members",
            targetText: null,
            needsMessageEvidence: false,
            answerConfidence: "best_effort",
            confidence: 0.55,
            reason: "Need another member page to cover the requested scope.",
        };
    }

    if (questionIntent === "person_identity" && profile) {
        return {
            source: "deterministic",
            questionIntent,
            routeIntent,
            nextAction: "answer",
            targetText: profile.displayName,
            authorId: profile.id,
            authorQuery: profile.username,
            resolvedPerson: {
                id: profile.id,
                username: profile.username,
                displayName: profile.displayName,
                globalName: profile.globalName,
                nickname: profile.nickname,
                roles: profile.roles,
            },
            needsMessageEvidence: false,
            answerConfidence: "confident",
            confidence: 0.7,
            reason: "Resolved member profile already answers the identity question.",
        };
    }

    if (
        (questionIntent === "person_messages" || questionIntent === "channel_or_topic_search") &&
        hasStrongMessageEvidence
    ) {
        return {
            source: "deterministic",
            questionIntent,
            routeIntent,
            nextAction: "answer",
            targetText: likelyTarget,
            authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
            authorQuery: likelyTarget ?? undefined,
            topicText,
            channelHintText: topicText,
            channelIds: targetChannelIds.length ? targetChannelIds : undefined,
            searchQuery: topicText ?? input.question,
            needsMessageEvidence,
            answerConfidence: "confident",
            confidence: 0.72,
            reason: "Strong message evidence is already available.",
        };
    }

    if (
        questionIntent === "member_list_or_ordinal" &&
        memberEvidence &&
        !memberEvidenceNeedsMore(input.question, memberEvidence, ordinal)
    ) {
        return {
            source: "deterministic",
            questionIntent,
            routeIntent,
            nextAction: "answer",
            targetText: null,
            needsMessageEvidence: false,
            answerConfidence: "confident",
            confidence: 0.65,
            reason: "Available member evidence covers the requested scope.",
        };
    }

    if (!input.toolRuns.length) {
        if (questionIntent === "server_context") {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "get_guild_context",
                targetText: null,
                needsMessageEvidence: false,
                answerConfidence: "best_effort",
                confidence: 0.4,
                reason: "Start with live guild context.",
            };
        }

        if (questionIntent === "member_list_or_ordinal") {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "list_members",
                targetText: likelyTarget,
                searchQuery: likelyTarget ?? input.question,
                needsMessageEvidence: false,
                answerConfidence: "best_effort",
                confidence: 0.4,
                reason: "Start with live member listing.",
            };
        }

        if (questionIntent === "person_identity") {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "get_member_profile",
                targetText: likelyTarget,
                authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                authorQuery: likelyTarget ?? undefined,
                needsMessageEvidence: false,
                answerConfidence: "best_effort",
                confidence: 0.4,
                reason: "Start by resolving the member profile.",
            };
        }

        if (targetChannelIds.length) {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "search_messages",
                targetText: likelyTarget,
                authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                authorQuery: likelyTarget ?? undefined,
                channelIds: targetChannelIds,
                topicText,
                channelHintText: topicText,
                searchQuery: topicText ?? input.question,
                needsMessageEvidence,
                answerConfidence: "best_effort",
                confidence: 0.45,
                reason: "Start with a scoped search in the routed channel.",
            };
        }

        return {
            source: "deterministic",
            questionIntent,
            routeIntent,
            nextAction: "search_messages",
            targetText: likelyTarget,
            authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
            authorQuery: likelyTarget ?? undefined,
            channelIds: targetChannelIds.length ? targetChannelIds : undefined,
            topicText,
            channelHintText: topicText,
            searchQuery: topicText ?? input.question,
            needsMessageEvidence,
            answerConfidence: "best_effort",
            confidence: 0.35,
            reason: "Start with local memory search.",
        };
    }

    if (
        questionIntent === "person_identity" &&
        latestSearchRun &&
        groundingSummary.messageEvidenceCount > 0 &&
        !profile
    ) {
        return {
            source: "deterministic",
            questionIntent,
            routeIntent,
            nextAction: "best_effort_answer",
            targetText: likelyTarget,
            authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
            authorQuery: likelyTarget ?? undefined,
            topicText,
            channelHintText: topicText,
            searchQuery: topicText ?? input.question,
            needsMessageEvidence: false,
            answerConfidence: "best_effort",
            confidence: 0.5,
            reason: "There is some identity-adjacent evidence, but no resolved profile yet.",
        };
    }

    if (
        (questionIntent === "person_messages" ||
            questionIntent === "channel_or_topic_search" ||
            questionIntent === "broad_search") &&
        latestSearchRun
    ) {
        if (hasStrongMessageEvidence) {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "answer",
                targetText: likelyTarget,
                authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                authorQuery: likelyTarget ?? undefined,
                channelIds: targetChannelIds.length ? targetChannelIds : undefined,
                topicText,
                channelHintText: topicText,
                searchQuery: topicText ?? input.question,
                needsMessageEvidence,
                answerConfidence: "confident",
                confidence: 0.72,
                reason: "Message evidence already answers the question.",
            };
        }

        const nextTargetChannelId = targetChannelIds.find(
            (channelId) => !input.context.crawledChannelIds.has(channelId)
        );
        if (nextTargetChannelId) {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "crawl_channel_messages",
                targetText: likelyTarget,
                authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                authorQuery: likelyTarget ?? undefined,
                channelIds: [nextTargetChannelId],
                topicText,
                channelHintText: topicText,
                searchQuery: topicText ?? input.question,
                needsMessageEvidence,
                answerConfidence: hasAnyEvidence ? "best_effort" : "insufficient",
                confidence: 0.55,
                reason: "Local search is weak; crawl the hinted channel next.",
            };
        }

        if (!latestChannelListRun && !targetChannelIds.length) {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "list_relevant_channels",
                targetText: likelyTarget,
                authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                authorQuery: likelyTarget ?? undefined,
                topicText,
                channelHintText: topicText,
                searchQuery: topicText ?? input.question,
                needsMessageEvidence,
                answerConfidence: hasAnyEvidence ? "best_effort" : "insufficient",
                confidence: 0.48,
                reason: "Need likely channel candidates before giving up.",
            };
        }

        if (latestChannelListRun) {
            const nextChannel = pickNextChannelCandidate(
                latestChannelListRun.data as ChannelCandidate[] | null,
                input.context
            );
            if (nextChannel) {
                return {
                    source: "deterministic",
                    questionIntent,
                    routeIntent,
                    nextAction: "crawl_channel_messages",
                    targetText: nextChannel.channelName,
                    authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                    authorQuery: likelyTarget ?? undefined,
                    channelIds: [nextChannel.channelId],
                    topicText,
                    channelHintText: nextChannel.channelName,
                    searchQuery: topicText ?? input.question,
                    needsMessageEvidence,
                    answerConfidence: hasAnyEvidence ? "best_effort" : "insufficient",
                    confidence: 0.52,
                    reason: "Crawl the strongest uncrawled channel candidate.",
                };
            }
        }

        if (latestCrawlRun) {
            if (latestCrawlPreviewCount > 0) {
                return {
                    source: "deterministic",
                    questionIntent,
                    routeIntent,
                    nextAction: "best_effort_answer",
                    targetText: likelyTarget,
                    authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                    authorQuery: likelyTarget ?? undefined,
                    channelIds: targetChannelIds.length ? targetChannelIds : undefined,
                    topicText,
                    channelHintText: topicText,
                    searchQuery: topicText ?? input.question,
                    needsMessageEvidence,
                    answerConfidence: "best_effort",
                    confidence: 0.58,
                    reason: "The live crawl already found useful preview messages.",
                };
            }

            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "search_messages",
                targetText: likelyTarget,
                authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                authorQuery: likelyTarget ?? undefined,
                channelIds: targetChannelIds.length ? targetChannelIds : undefined,
                topicText,
                channelHintText: topicText,
                searchQuery: topicText ?? input.question,
                needsMessageEvidence,
                answerConfidence: hasAnyEvidence ? "best_effort" : "insufficient",
                confidence: 0.5,
                reason: "Retry local search after crawling.",
            };
        }

        if (hasAnyEvidence) {
            return {
                source: "deterministic",
                questionIntent,
                routeIntent,
                nextAction: "best_effort_answer",
                targetText: likelyTarget,
                authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
                authorQuery: likelyTarget ?? undefined,
                channelIds: targetChannelIds.length ? targetChannelIds : undefined,
                topicText,
                channelHintText: topicText,
                searchQuery: topicText ?? input.question,
                needsMessageEvidence,
                answerConfidence: "best_effort",
                confidence: 0.45,
                reason: "Some evidence exists, but not enough for a confident answer.",
            };
        }
    }

    return {
        source: "deterministic",
        questionIntent,
        routeIntent,
        nextAction: "search_messages",
        targetText: likelyTarget,
        authorId: overrides.authorIds[0] ?? input.priorContext?.authorId ?? undefined,
        authorQuery: likelyTarget ?? undefined,
        channelIds: targetChannelIds.length ? targetChannelIds : undefined,
        topicText,
        channelHintText: topicText,
        searchQuery: topicText ?? input.question,
        needsMessageEvidence,
        answerConfidence: groundingSummary.messageEvidenceCount > 0 ? "best_effort" : "insufficient",
        confidence: 0.3,
        reason: "Fallback controller decision.",
    };
}

function pickNextChannelCandidate(
    channels: ChannelCandidate[] | null,
    context: SearchContext
): ChannelCandidate | null {
    if (!channels?.length) {
        return null;
    }

    return (
        channels.find((channel) => !context.crawledChannelIds.has(channel.channelId)) || null
    );
}

function normalizeControllerDecision(
    raw: RetrievalControllerModelDecision,
    fallback: RetrievalControllerDecision,
    input: RetrievalControllerInput,
    readableChannels: Array<{ id: string; name: string }>,
    usedModel: boolean
): RetrievalControllerDecision {
    const questionIntent = raw.questionIntent ?? fallback.questionIntent;
    const routeIntent = mapQuestionIntentToRouteIntent(questionIntent);
    const searchQuery = (raw.searchQuery || raw.topicText || fallback.searchQuery || "").trim();
    const targetText = (raw.targetText || fallback.targetText || "").trim() || null;
    const topicText = (raw.topicText || fallback.topicText || "").trim() || null;
    const channelHintText =
        (raw.channelHintText || fallback.channelHintText || topicText || "").trim() || null;
    const channelIds = normalizeChannelIds(
        raw.channelIds,
        fallback.channelIds,
        channelHintText,
        readableChannels
    );
    const nextAction = normalizeNextAction(
        raw.nextAction,
        fallback.nextAction,
        input.forcedFinal ?? false
    );
    const answerConfidence = normalizeAnswerConfidence(
        raw.answerConfidence,
        fallback.answerConfidence,
        nextAction
    );
    const authorId = raw.authorId || fallback.authorId;
    const authorQuery =
        (raw.authorQuery ||
            fallback.authorQuery ||
            targetText ||
            input.priorContext?.authorQuery ||
            "")?.trim() || undefined;

    return {
        source: usedModel ? "ai" : fallback.source,
        questionIntent,
        routeIntent,
        nextAction,
        targetText,
        authorId,
        authorQuery,
        channelIds: channelIds.length ? channelIds : undefined,
        topicText,
        channelHintText,
        resolvedPerson: input.priorContext?.resolvedPerson ?? fallback.resolvedPerson ?? null,
        searchQuery: searchQuery || fallback.searchQuery || input.question,
        needsMessageEvidence:
            typeof raw.needsMessageEvidence === "boolean"
                ? raw.needsMessageEvidence
                : fallback.needsMessageEvidence,
        answerConfidence,
        confidence:
            typeof raw.confidence === "number"
                ? Math.max(0, Math.min(1, raw.confidence))
                : fallback.confidence,
        reason: raw.reason?.trim() || fallback.reason,
    };
}

function normalizeChannelIds(
    raw: string[] | undefined,
    fallback: string[] | undefined,
    channelHintText: string | null,
    readableChannels: Array<{ id: string; name: string }>
): string[] {
    const explicit = raw?.filter(Boolean) ?? fallback?.filter(Boolean) ?? [];
    if (explicit.length) {
        return [...new Set(explicit)];
    }

    if (!channelHintText) {
        return [];
    }

    const normalizedHint = normalizeQuestion(channelHintText).replace(/[^a-z0-9]/g, "");
    return readableChannels
        .filter((channel) => {
            const normalizedName = normalizeQuestion(channel.name).replace(/[^a-z0-9]/g, "");
            return normalizedName === normalizedHint;
        })
        .map((channel) => channel.id);
}

function normalizeNextAction(
    raw: RetrievalControllerModelDecision["nextAction"],
    fallback: RetrievalControllerDecision["nextAction"],
    forcedFinal: boolean
): RetrievalControllerDecision["nextAction"] {
    const allowed = new Set<RetrievalControllerDecision["nextAction"]>([
        "answer",
        "best_effort_answer",
        "search_messages",
        "get_member_profile",
        "list_members",
        "get_guild_context",
        "list_relevant_channels",
        "crawl_channel_messages",
    ]);
    if (raw && allowed.has(raw)) {
        return raw;
    }

    if (forcedFinal && !fallback) {
        return "answer";
    }

    return fallback;
}

function normalizeAnswerConfidence(
    raw: GroundedAnswerMode | undefined,
    fallback: GroundedAnswerMode,
    nextAction: RetrievalControllerDecision["nextAction"]
): GroundedAnswerMode {
    if (raw === "confident" || raw === "best_effort" || raw === "insufficient") {
        return raw;
    }

    if (nextAction === "best_effort_answer") {
        return "best_effort";
    }

    return fallback;
}

function summarizeToolHistory(toolRuns: DiscordToolResult[]): string {
    if (!toolRuns.length) {
        return "none";
    }

    return toolRuns
        .map((run, index) => {
            const parts = [`${index + 1}. ${run.tool}: ${run.summary}`];
            if (run.tool === "search_messages") {
                const items = Array.isArray(run.data) ? run.data.slice(0, 3) : [];
                items.forEach((item: any, itemIndex) => {
                    parts.push(
                        `   - hit ${itemIndex + 1}: [${item.channelName}] ${item.authorName}: ${String(item.content || "").slice(0, 160)}`
                    );
                });
            } else if (run.tool === "get_member_profile" && run.data) {
                const profile = run.data as any;
                parts.push(
                    `   - profile: ${profile.displayName} (@${profile.username}) roles=${(profile.roles || []).join(", ") || "none"}`
                );
            } else if (run.tool === "list_relevant_channels") {
                const channels = Array.isArray(run.data) ? (run.data as ChannelCandidate[]).slice(0, 5) : [];
                channels.forEach((channel) => {
                    parts.push(
                        `   - channel: ${channel.channelName} (${channel.channelId}) indexed=${channel.isIndexed ? "yes" : "no"}`
                    );
                });
            } else if (run.tool === "crawl_channel_messages" && run.data) {
                const crawl = run.data as any;
                parts.push(
                    `   - crawl: ${crawl.channelName} fetched=${crawl.messagesFetched} stored=${crawl.messagesStored}`
                );
            } else if (run.tool === "list_members" && run.data) {
                const members = run.data as any;
                parts.push(
                    `   - members: returned=${members.returnedCount} total=${members.totalCount} hasMore=${members.hasMore ? "yes" : "no"}`
                );
            } else if (run.tool === "get_guild_context" && run.data) {
                const context = run.data as any;
                parts.push(
                    `   - guild: ${context.name} members=${context.memberCount} channels=${context.channelCount}`
                );
            }
            if (run.errorMessage) {
                parts.push(`   - error: ${run.errorMessage}`);
            }
            return parts.join("\n");
        })
        .join("\n");
}

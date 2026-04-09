import type { Guild } from "discord.js";
import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";
import { DiscordChannelCrawlService } from "@/discord/live/DiscordChannelCrawlService";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import type {
    ConversationResolutionContext,
    RouteDecision,
    RouteIntent,
} from "@/shared/appTypes";
import {
    extractTopicHint,
    extractLikelyPersonName,
    extractMentionedChannelIds,
    extractMentionedUserIds,
    isGuildContextQuestion,
    isIdentityQuestion,
    isMemberDiscoveryQuestion,
    isPersonMessageQuestion,
    isReferentialFollowUp,
    needsAiRouting,
} from "@/agent/orchestration/questionAnalysis";

type RouterInput = {
    question: string;
    guild: Guild | null;
    currentChannelId?: string | null;
    priorContext?: ConversationResolutionContext | null;
};

type RouterModelDecision = {
    intent: RouteIntent;
    targetText?: string;
    topicText?: string;
    channelHintText?: string;
    confidence?: number;
    reason?: string;
};

export async function routeDiscordQuestion(
    input: RouterInput
): Promise<RouteDecision> {
    const deterministic = getDeterministicRouteDecision(input);
    if (deterministic) {
        return deterministic;
    }

    if (!needsAiRouting(input.question)) {
        return {
            source: "deterministic",
            intent: "broad_search",
            targetText: null,
            confidence: 1,
            reason: "No explicit or ambiguous target detected; use broad search.",
        };
    }

    return routeAmbiguousQuestion(input);
}

function getDeterministicRouteDecision(input: RouterInput): RouteDecision | null {
    const mentionedChannelIds = extractMentionedChannelIds(input.question);
    if (mentionedChannelIds.length) {
        return {
            source: "deterministic",
            intent: "channel_target",
            targetText: null,
            channelIds: mentionedChannelIds,
            confidence: 1,
            reason: "Question includes explicit channel mention(s).",
        };
    }

    const mentionedUserIds = extractMentionedUserIds(input.question);
    if (mentionedUserIds.length) {
        return {
            source: "deterministic",
            intent: "person_target",
            targetText: null,
            authorId: mentionedUserIds[0],
            confidence: 1,
            reason: "Question includes explicit user mention.",
        };
    }

    if (isMemberDiscoveryQuestion(input.question)) {
        return {
            source: "deterministic",
            intent: "member_lookup",
            targetText: null,
            confidence: 1,
            reason: "Question clearly asks about members or ordinal member positions.",
        };
    }

    if (isGuildContextQuestion(input.question)) {
        return {
            source: "deterministic",
            intent: "server_context",
            targetText: null,
            confidence: 1,
            reason: "Question clearly asks about the current server.",
        };
    }

    const followUpRoute = buildFollowUpRouteDecision(input);
    if (followUpRoute) {
        return followUpRoute;
    }

    return null;
}

async function routeAmbiguousQuestion(input: RouterInput): Promise<RouteDecision> {
    const channelHints = DiscordLiveService.listReadableGuildChannels(input.guild)
        .map((channel) => channel.name)
        .slice(0, 50)
        .join(", ");
    const fallbackTarget = extractLikelyPersonName(input.question) ?? "";
    const fallbackTopic = extractTopicHint(input.question);
    const fallback = inferFallbackRoute(input, fallbackTarget);

    const decision = await ModelGateway.generateJson<RouterModelDecision>(
        [
            { role: "system", content: "Return strict JSON only." },
            {
                role: "user",
                content: PromptRegistry.render("tasks/route_discord_intent", {
                    question: input.question,
                    current_channel_id: input.currentChannelId || "",
                    readable_channels: channelHints || "none",
                    prior_context: input.priorContext
                        ? JSON.stringify(input.priorContext, null, 2)
                        : "none",
                }),
            },
        ],
        {
            intent: fallback.intent,
            targetText: fallback.targetText || undefined,
            topicText: fallback.topicText || undefined,
            channelHintText: fallback.channelHintText || undefined,
            confidence: fallback.confidence,
            reason: fallback.reason,
        },
        {
            traceContext: {
                traceLabel: "discord_question_routing",
                questionPreview: input.question,
            },
        }
    );

    const intent = decision.intent ?? fallback.intent;
    const targetText = (decision.targetText || fallback.targetText || "").trim() || null;
    const topicText =
        (decision.topicText || fallbackTopic || fallback.topicText || "").trim() || null;
    const channelHintText =
        (decision.channelHintText ||
            fallback.channelHintText ||
            topicText ||
            (intent === "channel_target" ? targetText : ""))?.trim() || null;
    const resolvedChannelIds = resolveChannelHints(
        input.guild,
        intent,
        targetText,
        channelHintText,
        input.currentChannelId
    );

    return {
        source: "ai",
        intent,
        targetText,
        channelIds: resolvedChannelIds?.length ? resolvedChannelIds : undefined,
        authorId:
            intent === "person_target"
                ? input.priorContext?.authorId ?? undefined
                : undefined,
        authorQuery:
            intent === "person_target"
                ? targetText || input.priorContext?.authorQuery || undefined
                : undefined,
        topicText,
        channelHintText,
        resolvedPerson:
            intent === "person_target"
                ? input.priorContext?.resolvedPerson ?? null
                : null,
        confidence: normalizeConfidence(decision.confidence, fallback.confidence),
        reason: decision.reason?.trim() || fallback.reason,
    };
}

function inferFallbackRoute(input: RouterInput, fallbackTarget: string): RouteDecision {
    const normalized = input.question.toLowerCase();
    const topicText = extractTopicHint(input.question);
    if (normalized.includes("canal") || normalized.includes("channel")) {
        return {
            source: "deterministic",
            intent: "channel_target",
            targetText: fallbackTarget || null,
            topicText,
            channelHintText: fallbackTarget || topicText || null,
            confidence: 0.45,
            reason: "Question language suggests a channel target.",
        };
    }

    if (
        normalized.includes("perfil") ||
        normalized.includes("profile") ||
        normalized.includes("bio")
    ) {
        return {
            source: "deterministic",
            intent: "person_target",
            targetText: fallbackTarget || null,
            topicText,
            channelHintText: topicText,
            authorId: input.priorContext?.authorId ?? undefined,
            authorQuery: fallbackTarget || input.priorContext?.authorQuery || undefined,
            resolvedPerson: input.priorContext?.resolvedPerson ?? null,
            confidence: 0.45,
            reason: "Question language suggests a person/profile target.",
        };
    }

    return {
        source: "deterministic",
        intent: "broad_search",
        targetText: null,
        topicText,
        channelHintText: topicText,
        confidence: 0.35,
        reason: "Ambiguous question; default to broad search.",
    };
}

function buildFollowUpRouteDecision(
    input: RouterInput
): RouteDecision | null {
    const prior = input.priorContext;
    if (!prior?.resolvedPerson || !isReferentialFollowUp(input.question)) {
        return null;
    }

    const topicText = extractTopicHint(input.question);
    const channelHintText = topicText || prior.channelHintText || null;
    const channelIds = resolveChannelHints(
        input.guild,
        "person_target",
        null,
        channelHintText,
        input.currentChannelId
    );

    if (isPersonMessageQuestion(input.question) || topicText) {
        return {
            source: "deterministic",
            intent: "person_target",
            targetText: prior.targetText || prior.resolvedPerson.displayName,
            authorId: prior.authorId || prior.resolvedPerson.id,
            authorQuery: prior.authorQuery || prior.resolvedPerson.username,
            topicText,
            channelHintText,
            channelIds: channelIds?.length ? channelIds : prior.channelIds,
            resolvedPerson: prior.resolvedPerson,
            confidence: 0.9,
            reason: "Follow-up question reuses the previously resolved person target.",
        };
    }

    if (isIdentityQuestion(input.question)) {
        return {
            source: "deterministic",
            intent: "person_target",
            targetText: prior.targetText || prior.resolvedPerson.displayName,
            authorId: prior.authorId || prior.resolvedPerson.id,
            authorQuery: prior.authorQuery || prior.resolvedPerson.username,
            resolvedPerson: prior.resolvedPerson,
            confidence: 0.9,
            reason: "Follow-up question still refers to the previously resolved person.",
        };
    }

    return null;
}

function resolveChannelHints(
    guild: Guild | null,
    intent: RouteIntent,
    targetText: string | null,
    channelHintText: string | null,
    currentChannelId?: string | null
): string[] | undefined {
    const lookup =
        intent === "channel_target"
            ? targetText
            : channelHintText;
    if (!lookup) {
        return undefined;
    }

    const resolved = DiscordChannelCrawlService.resolveChannelIdsByName(
        guild,
        lookup,
        currentChannelId
    );
    return resolved.length ? resolved : undefined;
}

function normalizeConfidence(
    value: number | undefined,
    fallback: number
): number {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return fallback;
    }

    return Math.max(0, Math.min(1, value));
}

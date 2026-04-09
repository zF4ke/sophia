import { createQuestionFingerprint } from "@/agent/orchestration/questionFingerprint";
import { getRequestCacheContext } from "@/agent/orchestration/requestCacheContext";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { RouteDecision } from "@/shared/appTypes";
import type { GroundingDecision, GroundingAssessment, ReusableGroundedContext } from "@/agent/orchestration/types";

const REUSABLE_CONTEXT_TTL_MS = 20 * 60_000;
const MAX_REUSABLE_CONTEXT_AGE_RESPONSES = 6;

export function findReusableGroundedContext(options: {
    guildId: string | null;
    currentChannelId: string | null;
    question: string;
    routeDecision: RouteDecision;
    requireSufficient?: boolean;
}): ReusableGroundedContext | null {
    const questionFingerprint = createQuestionFingerprint(options.question);
    if (!questionFingerprint) {
        return null;
    }

    return DiscordMemoryService.getReusableGroundedContext({
        guildId: options.guildId,
        currentChannelId: options.currentChannelId,
        questionFingerprint,
        routeIntent: options.routeDecision.intent,
        requireSufficient: options.requireSufficient,
        currentResponseOrdinal: getRequestCacheContext().responseOrdinal,
        maxResponsesAgo: MAX_REUSABLE_CONTEXT_AGE_RESPONSES,
    });
}

export function saveReusableGroundedContext(options: {
    guildId: string | null;
    currentChannelId: string | null;
    question: string;
    routeDecision: RouteDecision;
    grounding: GroundingAssessment;
    groundingDecision: GroundingDecision;
    toolRuns: ReusableGroundedContext["toolRuns"];
}): void {
    const questionFingerprint = createQuestionFingerprint(options.question);
    if (!questionFingerprint) {
        return;
    }

    const hasReusableMaterial =
        options.toolRuns.length > 0 &&
        (
            options.grounding.summary.messageEvidenceCount > 0 ||
            options.grounding.summary.liveEvidenceCount > 0 ||
            options.grounding.evidence.trim().length > 0
        );

    if (!hasReusableMaterial) {
        return;
    }

    const now = Date.now();
    const requestContext = getRequestCacheContext();
    DiscordMemoryService.saveReusableGroundedContext({
        guildId: options.guildId,
        channelId: options.currentChannelId,
        channelScopeKey: buildChannelScopeKey(options.currentChannelId),
        questionFingerprint,
        routeIntent: options.routeDecision.intent,
        evidenceText: options.grounding.evidence,
        citations: options.grounding.citations,
        toolRuns: options.toolRuns,
        sufficient: options.groundingDecision.sufficient,
        groundingDecisionMode: options.groundingDecision.mode,
        createdTimestamp: now,
        expiryTimestamp: now + REUSABLE_CONTEXT_TTL_MS,
        createdResponseOrdinal: requestContext.responseOrdinal,
    });
}

function buildChannelScopeKey(channelId: string | null): string {
    return channelId || "__guild__";
}

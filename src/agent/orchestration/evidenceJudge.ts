import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";
import { extractBestChunk } from "@/discord/tools/runtime/resultReaders";
import type {
    DiscordToolResult,
    EvidenceJudgeResult,
    GroundingSummary,
    RouteDecision,
} from "@/shared/appTypes";
import {
    isMemberDiscoveryQuestion,
    isPersonMessageQuestion,
} from "@/agent/orchestration/questionAnalysis";
import type { GroundingAssessment, GroundingDecision } from "@/agent/orchestration/types";

const STRONG_SEARCH_SCORE = 0.45;

type GroundingPayload = {
    summary: GroundingSummary;
    evidence: string;
};

export async function decideGroundingSufficiency(options: {
    question: string;
    grounding: GroundingPayload;
    toolRuns: DiscordToolResult[];
    routeDecision: RouteDecision;
    reuseMode?: "reused" | null;
}): Promise<GroundingDecision> {
    if (options.reuseMode === "reused") {
        return {
            sufficient: true,
            mode: "reused",
            answerMode: "confident",
            reason: "A reusable grounded context was found.",
            missingInformation: null,
        };
    }

    const heuristic = getHeuristicGroundingDecision(
        options.question,
        options.grounding,
        options.toolRuns,
        options.routeDecision
    );
    if (heuristic) {
        return heuristic;
    }

    const judged = await judgeGroundingSufficiency(
        options.question,
        options.grounding
    );
    return {
        sufficient: judged.sufficient,
        mode: "judge",
        answerMode: judged.sufficient ? "confident" : "insufficient",
        reason: judged.reason,
        missingInformation: judged.missingInformation,
    };
}

export function getHeuristicGroundingDecision(
    question: string,
    grounding: GroundingPayload,
    toolRuns: DiscordToolResult[],
    routeDecision: RouteDecision
): GroundingDecision | null {
    if (
        grounding.summary.messageEvidenceCount <= 0 &&
        grounding.summary.liveEvidenceCount <= 0
    ) {
        return {
            sufficient: false,
            mode: "heuristic",
            answerMode: "insufficient",
            reason: "No direct evidence is available yet.",
            missingInformation: "More Discord evidence is needed.",
        };
    }

    if (
        routeDecision.intent === "server_context" &&
        grounding.summary.liveEvidenceCount > 0
    ) {
        return {
            sufficient: true,
            mode: "heuristic",
            answerMode: "confident",
            reason: "Live guild context directly answers the question.",
            missingInformation: null,
        };
    }

    if (
        isMemberDiscoveryQuestion(question) &&
        grounding.summary.liveEvidenceCount > 0
    ) {
        if (grounding.summary.sufficient) {
            return {
                sufficient: true,
                mode: "heuristic",
                answerMode: "confident",
                reason: "Member evidence already covers the requested member scope.",
                missingInformation: null,
            };
        }

        return {
            sufficient: false,
            mode: "heuristic",
            answerMode: "insufficient",
            reason: "Member evidence is still incomplete for the requested scope.",
            missingInformation: "More member evidence is needed.",
        };
    }

    if (
        routeDecision.intent === "person_target" &&
        (routeDecision.topicText || isPersonMessageQuestion(question)) &&
        grounding.summary.messageEvidenceCount <= 0
    ) {
        return {
            sufficient: false,
            mode: "heuristic",
            answerMode: "insufficient",
            reason: "Profile or member evidence alone does not answer what that person said.",
            missingInformation: "Need message evidence from that person on the requested topic.",
        };
    }

    const bestChunk = extractBestChunk(toolRuns);
    if (
        bestChunk &&
        grounding.summary.messageEvidenceCount > 0 &&
        bestChunk.totalScore >= STRONG_SEARCH_SCORE
    ) {
        return {
            sufficient: true,
            mode: "heuristic",
            answerMode: "confident",
            reason: "Message evidence is already strong enough.",
            missingInformation: null,
        };
    }

    return null;
}

export async function decideGroundingFromAssessment(options: {
    question: string;
    assessment: GroundingAssessment;
    toolRuns: DiscordToolResult[];
    routeDecision: RouteDecision;
    reuseMode?: "reused" | null;
}): Promise<GroundingDecision> {
    return decideGroundingSufficiency({
        question: options.question,
        grounding: {
            summary: options.assessment.summary,
            evidence: options.assessment.evidence,
        },
        toolRuns: options.toolRuns,
        routeDecision: options.routeDecision,
        reuseMode: options.reuseMode,
    });
}

async function judgeGroundingSufficiency(
    question: string,
    grounding: GroundingPayload
): Promise<EvidenceJudgeResult> {
    const fallback: EvidenceJudgeResult = {
        sufficient: grounding.summary.sufficient,
        reason: grounding.summary.sufficient
            ? "Heuristic grounding check found enough evidence."
            : "Heuristic grounding check did not find enough evidence.",
        missingInformation: grounding.summary.sufficient
            ? null
            : "More Discord evidence is needed.",
    };

    let result: EvidenceJudgeResult | undefined;
    try {
        result = await ModelGateway.generateJson<EvidenceJudgeResult>(
            [
                { role: "system", content: "Return strict JSON only." },
                {
                    role: "user",
                    content: PromptRegistry.render("tasks/judge_grounding_sufficiency", {
                        question,
                        message_evidence_count: grounding.summary.messageEvidenceCount,
                        live_evidence_count: grounding.summary.liveEvidenceCount,
                        evidence: grounding.evidence,
                    }),
                },
            ],
            fallback,
            {
                traceContext: {
                    traceLabel: "grounding_sufficiency_judgment",
                    questionPreview: question,
                },
            }
        );
    } catch {
        return fallback;
    }

    if (!result || typeof result.sufficient !== "boolean") {
        return fallback;
    }

    return {
        sufficient: result.sufficient,
        reason: typeof result.reason === "string" ? result.reason.trim() : fallback.reason,
        missingInformation:
            typeof result.missingInformation === "string"
                ? result.missingInformation.trim()
                : result.missingInformation === null
                  ? null
                  : fallback.missingInformation,
    };
}

import type {
    ChannelContextMessage,
    EvidenceItem,
    RuntimeMode,
} from "@/runtime/contracts";
import type { GroundedAnswerMode, RequestClassification } from "@/shared/appTypes";

export function classify(mode: RuntimeMode): RequestClassification {
    return {
        mode: mode === "research" ? "discord_grounded" : "direct_answer",
        reason:
            mode === "research"
                ? "The runtime routed this request through Discord retrieval."
                : "The runtime can answer without Discord retrieval.",
    };
}

export function countEvidence(state: { evidence: EvidenceItem[] }) {
    return state.evidence.reduce(
        (acc, item) => {
            if (
                item.evidenceRole === "message_evidence" ||
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

export function answerConfidenceForInsufficient(
    state: { evidence: EvidenceItem[] }
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

export function formatEvidenceTime(timestamp: number | null | undefined): string {
    return typeof timestamp === "number" && Number.isFinite(timestamp) && Math.abs(timestamp) <= 8.64e15
        ? new Date(timestamp).toISOString() : "timestamp unavailable";
}

export function formatChannelContext(context: ChannelContextMessage[]): string {
    if (!context.length) {
        return "No recent channel messages available.";
    }
    return context.map((msg) => `[${formatEvidenceTime(msg.createdTimestamp)}] ${msg.authorName}: ${msg.content}`).join("\n");
}

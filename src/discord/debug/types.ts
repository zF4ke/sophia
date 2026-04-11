import type { GroundedAnswerMode, WebStatus } from "@/shared/appTypes";
import type {
    ConversationContext,
    ReplyContext,
    RetrievalSummary,
    RuntimeDebugSession,
    RuntimeMode,
    StopReason,
    TurnTrigger,
} from "@/runtime/contracts";

export type DebugSectionKey =
    | "request"
    | "conversation"
    | "retrieval"
    | "context"
    | "timeline";

export interface DebugTimelineEntry {
    label: string;
    detail: string;
    tone: "info" | "success" | "warning" | "error";
    timestamp: number;
}

export interface DebugTraceState {
    questionPreview: string;
    status: "running" | "completed" | "failed";
    stage: string;
    requesterLabel: string | null;
    trigger: TurnTrigger | null;
    classificationMode: "direct_answer" | "discord_grounded" | null;
    runtimeMode: RuntimeMode | null;
    selectedCapabilities: string[];
    toolCallCount: number;
    groundingSummary: {
        messageEvidenceCount: number;
        liveEvidenceCount: number;
        sufficient: boolean;
    } | null;
    retrievalSummary: RetrievalSummary | null;
    groundedAnswerMode: GroundedAnswerMode | null;
    stopReason: StopReason | null;
    checkpointThreadId: string | null;
    conversationContext: {
        threadId: string | null;
        kind: ConversationContext["kind"] | null;
        replyAnchorMessageId: string | null;
        replyContext: ReplyContext | null;
    };
    webStatus: WebStatus | null;
    contextPreview: {
        recentChannelMessages: string[];
        evidencePreview: string[];
        recentTurns: string[];
    } | null;
    recentEvents: string[];
    timeline: DebugTimelineEntry[];
    collapsedSections: Record<DebugSectionKey, boolean>;
    startedAt: number;
    failureMessage: string | null;
}

export type DebugSessionReporter = RuntimeDebugSession;

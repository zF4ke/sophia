import type { GroundedAnswerMode } from "@/shared/appTypes";
import type {
    RuntimeDebugSession,
    RuntimeMode,
    StopReason,
    TurnTrigger,
} from "@/runtime/contracts";

export interface DebugTimelineEntry {
    label: string;
    detail: string;
    tone: "info" | "success" | "warning" | "error";
    timestamp: number;
}

export interface NoteSnapshotEntry {
    seq: number;
    label: string | null;
    bodyPreview: string;
    wordCount: number;
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
    evidenceCount: number;
    confidence: GroundedAnswerMode | null;
    stopReason: StopReason | null;
    stopDetail: string | null;
    conversationThreadId: string | null;
    timeline: DebugTimelineEntry[];
    startedAt: number;
    failureMessage: string | null;
    cumulativePromptTokens: number;
    cumulativeCompletionTokens: number;
    contextUsagePercent: number | null;
    notesSnapshot: NoteSnapshotEntry[];
    planPreview: string | null;
}

export type DebugSessionReporter = RuntimeDebugSession;

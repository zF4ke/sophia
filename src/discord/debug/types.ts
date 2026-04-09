import type {
    GroundedAnswerMode,
    GroundingDecisionMode,
    GroundingSummary,
    RetrievalControllerDecision,
} from "@/shared/appTypes";

export interface DebugTraceState {
    questionPreview: string;
    status: "running" | "completed" | "failed";
    stage: string;
    mode: string | null;
    controllerDecision: RetrievalControllerDecision | null;
    toolNames: string[];
    groundingSummary: GroundingSummary | null;
    groundingDecisionMode: GroundingDecisionMode | null;
    groundedAnswerMode: GroundedAnswerMode | null;
    contextCacheStatus: "none" | "seeded" | "reused";
    recentEvents: string[];
    startedAt: number;
}

export interface DebugSessionReporter {
    setClassifying(): Promise<void>;
    setClassification(mode: "direct_answer" | "discord_grounded"): Promise<void>;
    setRouting?(decision: RetrievalControllerDecision): Promise<void>;
    setContextCacheStatus?(status: "none" | "seeded" | "reused"): Promise<void>;
    setPlanning(step: number): Promise<void>;
    setToolRunning(toolName: string, details?: string[]): Promise<void>;
    setToolProgress?(toolName: string, summary: string): Promise<void>;
    setToolResult(toolName: string, summary: string, itemCount?: number): Promise<void>;
    setGroundingSummary(
        summary: GroundingSummary,
        decisionMode?: GroundingDecisionMode,
        answerMode?: GroundedAnswerMode
    ): Promise<void>;
    setGenerating(): Promise<void>;
    finishSuccess(summary?: string): Promise<void>;
    finishError(error: unknown): Promise<void>;
}

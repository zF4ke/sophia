import type {
    GroundingDecisionMode,
    GroundingSummary,
    RouteDecision,
} from "@/shared/appTypes";

export interface DebugTraceState {
    questionPreview: string;
    status: "running" | "completed" | "failed";
    stage: string;
    mode: string | null;
    routeDecision: RouteDecision | null;
    toolNames: string[];
    groundingSummary: GroundingSummary | null;
    groundingDecisionMode: GroundingDecisionMode | null;
    contextCacheStatus: "none" | "seeded" | "reused";
    recentEvents: string[];
    startedAt: number;
}

export interface DebugSessionReporter {
    setClassifying(): Promise<void>;
    setClassification(mode: "direct_answer" | "discord_grounded"): Promise<void>;
    setRouting?(decision: RouteDecision): Promise<void>;
    setContextCacheStatus?(status: "none" | "seeded" | "reused"): Promise<void>;
    setPlanning(step: number): Promise<void>;
    setToolRunning(toolName: string, details?: string[]): Promise<void>;
    setToolResult(toolName: string, summary: string, itemCount?: number): Promise<void>;
    setGroundingSummary(
        summary: GroundingSummary,
        decisionMode?: GroundingDecisionMode
    ): Promise<void>;
    setGenerating(): Promise<void>;
    finishSuccess(summary?: string): Promise<void>;
    finishError(error: unknown): Promise<void>;
}

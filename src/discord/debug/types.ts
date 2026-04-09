import type { GroundingSummary } from "@/shared/appTypes";

export interface DebugTraceState {
    questionPreview: string;
    status: "running" | "completed" | "failed";
    stage: string;
    mode: string | null;
    toolNames: string[];
    groundingSummary: GroundingSummary | null;
    recentEvents: string[];
    startedAt: number;
}

export interface DebugSessionReporter {
    setClassifying(): Promise<void>;
    setClassification(mode: "direct_answer" | "discord_grounded"): Promise<void>;
    setPlanning(step: number): Promise<void>;
    setToolRunning(toolName: string, details?: string[]): Promise<void>;
    setToolResult(toolName: string, summary: string, itemCount?: number): Promise<void>;
    setGroundingSummary(summary: GroundingSummary): Promise<void>;
    setGenerating(): Promise<void>;
    finishSuccess(summary?: string): Promise<void>;
    finishError(error: unknown): Promise<void>;
}

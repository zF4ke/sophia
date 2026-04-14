import { Message, MessageFlags } from "discord.js";
import { renderDebugTrace } from "@/discord/debug/renderDebugTrace";
import type {
    DebugSessionReporter,
    DebugTimelineEntry,
    DebugTraceState,
} from "@/discord/debug/types";
import type { GroundedAnswerMode } from "@/shared/appTypes";
import type { RetrievalSummary, RuntimeMode, StopReason, TurnTrigger } from "@/runtime/contracts";

const MAX_PREVIEW_LENGTH = 140;
const MAX_TIMELINE = 30;

function normalizePreview(question: string): string {
    const compact = question.replace(/\s+/g, " ").trim();
    if (!compact) return "No text.";
    return compact.length > MAX_PREVIEW_LENGTH
        ? `${compact.slice(0, MAX_PREVIEW_LENGTH - 3)}...`
        : compact;
}

function normalizeError(error: unknown): string {
    if (error instanceof Error && error.message) return error.message;
    return "Unknown error";
}

function inferTone(event: string): DebugTimelineEntry["tone"] {
    const normalized = event.toLowerCase();
    if (normalized.includes("error") || normalized.includes("failed")) return "error";
    if (normalized.includes("insufficient") || normalized.includes("hits=0") || normalized.includes("fallback")) return "warning";
    if (normalized.includes("sufficient") || normalized.includes("completed")) return "success";
    return "info";
}

export class DebugSession implements DebugSessionReporter {
    private static readonly sessions = new Map<string, DebugSession>();
    private readonly state: DebugTraceState;
    private updateQueue: Promise<void> = Promise.resolve();

    public constructor(private readonly message: Message, question: string) {
        this.state = {
            questionPreview: normalizePreview(question),
            status: "running",
            stage: "Starting",
            requesterLabel: null,
            trigger: null,
            classificationMode: null,
            runtimeMode: null,
            selectedCapabilities: [],
            toolCallCount: 0,
            evidenceCount: 0,
            confidence: null,
            stopReason: null,
            stopDetail: null,
            conversationThreadId: null,
            timeline: [
                {
                    label: "Setup",
                    detail: "Debug session started.",
                    tone: "info",
                    timestamp: Date.now(),
                },
            ],
            startedAt: Date.now(),
            failureMessage: null,
            cumulativePromptTokens: 0,
            cumulativeCompletionTokens: 0,
            contextUsagePercent: null,
        };
        DebugSession.sessions.set(message.id, this);
    }

    public static getByMessageId(messageId: string): DebugSession | null {
        return this.sessions.get(messageId) || null;
    }

    public buildComponents() {
        return renderDebugTrace(this.state);
    }

    public async setClassifying(): Promise<void> {
        await this.mutate("Setup", "Ingesting turn");
    }

    public async setClassification(mode: "direct_answer" | "discord_grounded"): Promise<void> {
        await this.mutate("Agent loop", `Starting loop: ${mode}`, (state) => {
            state.classificationMode = mode;
        });
    }

    public async setRequesterContext(requesterLabel: string, trigger: TurnTrigger): Promise<void> {
        await this.mutate("Setup", `Requester: ${requesterLabel} via ${trigger}`, (state) => {
            state.requesterLabel = requesterLabel;
            state.trigger = trigger;
        });
    }

    public async setConversationContext(context: {
        threadId: string;
        kind: string | null;
        replyAnchorMessageId?: string | null;
        replyContext?: unknown;
    }): Promise<void> {
        await this.mutate("Setup", `Thread: ${context.threadId}`, (state) => {
            state.conversationThreadId = context.threadId;
        });
    }

    public async setRuntimeMode(mode: RuntimeMode): Promise<void> {
        await this.mutate("Evaluate", `Runtime mode: ${mode}`, (state) => {
            state.runtimeMode = mode;
        });
    }

    public async setCheckpointThread(threadId: string): Promise<void> {
        await this.mutate("Setup", `Thread: ${threadId}`, (state) => {
            state.conversationThreadId = threadId;
        });
    }

    public async setPlanning(step: number): Promise<void> {
        await this.mutate("Agent loop", `Loop iteration ${step}`);
    }

    public async setToolRunning(toolName: string, details: string[] = []): Promise<void> {
        await this.mutate(`Tool call`, `Running ${toolName}`, undefined, details);
    }

    public async setToolResult(toolName: string, summary: string, itemCount?: number): Promise<void> {
        await this.mutate(
            `Tool result`,
            itemCount === undefined ? `${toolName}: ${summary}` : `${toolName}: ${summary} (${itemCount})`,
            (state) => {
                state.toolCallCount += 1;
                if (!state.selectedCapabilities.includes(toolName)) {
                    state.selectedCapabilities.push(toolName);
                }
            }
        );
    }

    public async setToolProgress(toolName: string, summary: string): Promise<void> {
        await this.mutate(`Tool call`, `${toolName}: ${summary}`);
    }

    public async setRetrievalSummary(summary: RetrievalSummary): Promise<void> {
        const hits = summary.strongResultCount + summary.weakResultCount;
        await this.mutate(
            "Tool result",
            `origin=${summary.sourceOrigin}; hits=${hits}`,
            (state) => {
                state.evidenceCount = summary.accumulatedUniqueCount;
            }
        );
    }

    public async setEvidenceSummary(
        summary: { messageEvidenceCount: number; liveEvidenceCount: number; sufficient: boolean },
        _decisionMode?: string,
        answerMode?: GroundedAnswerMode
    ): Promise<void> {
        const total = summary.messageEvidenceCount + summary.liveEvidenceCount;
        await this.mutate(
            "Evaluate",
            summary.sufficient
                ? `Evidence sufficient (${total} items)`
                : `Evidence insufficient (${total} items)`,
            (state) => {
                state.evidenceCount = total;
                state.confidence = answerMode || null;
            }
        );
    }

    public async setStopReason(reason: StopReason, detail?: string | null): Promise<void> {
        await this.mutate(
            "Evaluate",
            detail ? `Stop: ${reason} — ${detail}` : `Stop: ${reason}`,
            (state) => {
                state.stopReason = reason;
                state.stopDetail = detail || null;
            }
        );
    }

    public async setConfidence(confidence: GroundedAnswerMode): Promise<void> {
        await this.mutate("Confidence update", `Confidence: ${confidence}`, (state) => {
            state.confidence = confidence;
        });
    }

    public async setTraceEvent(label: string, detail: string, timestamp?: number): Promise<void> {
        await this.mutate(label, detail, undefined, [], timestamp);
    }

    public async setGenerating(): Promise<void> {
        await this.mutate("Generate", "Producing the final answer");
    }

    public async setTokenUsage(promptTokens: number, completionTokens: number, contextUsagePercent: number | null): Promise<void> {
        const pct = contextUsagePercent != null ? ` (${contextUsagePercent.toFixed(1)}% context)` : "";
        await this.mutate(
            "Token usage",
            `${promptTokens.toLocaleString()} prompt + ${completionTokens.toLocaleString()} completion${pct}`,
            (state) => {
                state.cumulativePromptTokens = promptTokens;
                state.cumulativeCompletionTokens = completionTokens;
                state.contextUsagePercent = contextUsagePercent;
            }
        );
    }

    public async finishSuccess(summary = "Response completed"): Promise<void> {
        await this.mutate(
            "Done",
            summary,
            (state) => {
                state.status = "completed";
                state.stage = "Done";
            }
        );
    }

    public async finishError(error: unknown): Promise<void> {
        const message = normalizeError(error);
        await this.mutate(
            "Error",
            `Error: ${message}`,
            (state) => {
                state.status = "failed";
                state.stage = "Error";
                state.failureMessage = message;
            }
        );
    }

    private async mutate(
        stage: string,
        event: string,
        mutateState?: (state: DebugTraceState) => void,
        extraEvents: string[] = [],
        timestampOverride?: number
    ): Promise<void> {
        this.updateQueue = this.updateQueue
            .then(async () => {
                this.state.stage = stage;
                const normalizedExtraEvents = extraEvents
                    .map((item) => item.trim())
                    .filter(Boolean);
                const eventTimestamp = timestampOverride ?? Date.now();

                this.state.timeline = [
                    ...this.state.timeline,
                    {
                        label: stage,
                        detail: event,
                        tone: inferTone(event),
                        timestamp: eventTimestamp,
                    },
                    ...normalizedExtraEvents.map((detail) => ({
                        label: stage,
                        detail,
                        tone: inferTone(detail),
                        timestamp: eventTimestamp,
                    })),
                ]
                    .sort((left, right) => left.timestamp - right.timestamp)
                    .slice(-MAX_TIMELINE);

                mutateState?.(this.state);

                await this.message.edit({
                    components: this.buildComponents(),
                    flags: MessageFlags.IsComponentsV2,
                });
            })
            .catch((error) => {
                console.error("Error updating debug session:", error);
            });

        await this.updateQueue;
    }
}

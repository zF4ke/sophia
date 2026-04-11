import { Message, MessageFlags } from "discord.js";
import { renderDebugTrace } from "@/discord/debug/renderDebugTrace";
import type {
    DebugSectionKey,
    DebugSessionReporter,
    DebugTimelineEntry,
    DebugTraceState,
} from "@/discord/debug/types";
import type { GroundedAnswerMode, WebStatus } from "@/shared/appTypes";
import type { RetrievalSummary, RuntimeMode, StopReason, TurnTrigger } from "@/runtime/contracts";

const MAX_EVENTS = 8;
const MAX_PREVIEW_LENGTH = 140;
const MAX_TIMELINE = 25;
const DEFAULT_COLLAPSED_SECTIONS: Record<DebugSectionKey, boolean> = {
    request: false,
    conversation: false,
    retrieval: false,
    context: false,
    timeline: false,
};

function normalizePreview(question: string): string {
    const compact = question.replace(/\s+/g, " ").trim();
    if (!compact) {
        return "No text.";
    }

    return compact.length > MAX_PREVIEW_LENGTH
        ? `${compact.slice(0, MAX_PREVIEW_LENGTH - 3)}...`
        : compact;
}

function normalizeError(error: unknown): string {
    if (error instanceof Error && error.message) {
        return error.message;
    }

    return "Unknown error";
}

function inferTone(event: string): DebugTimelineEntry["tone"] {
    const normalized = event.toLowerCase();
    if (normalized.includes("error") || normalized.includes("failed")) {
        return "error";
    }
    if (
        normalized.includes("insufficient") ||
        normalized.includes("weak") ||
        normalized.includes("fallback") ||
        normalized.includes("plateau")
    ) {
        return "warning";
    }
    if (normalized.includes("sufficient") || normalized.includes("completed")) {
        return "success";
    }
    return "info";
}

export class DebugSession implements DebugSessionReporter {
    private static readonly sessions = new Map<string, DebugSession>();
    private static defaultCollapsedSections: Record<DebugSectionKey, boolean> = {
        ...DEFAULT_COLLAPSED_SECTIONS,
    };
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
            groundingSummary: null,
            retrievalSummary: null,
            groundedAnswerMode: null,
            stopReason: null,
            stopDetail: null,
            checkpointThreadId: null,
            conversationContext: {
                threadId: null,
                kind: null,
                replyAnchorMessageId: null,
                replyContext: null,
            },
            webStatus: null,
            contextPreview: null,
            recentEvents: ["Started"],
            timeline: [
                {
                    label: "start",
                    detail: "Debug session started.",
                    tone: "info",
                    timestamp: Date.now(),
                },
            ],
            collapsedSections: DebugSession.getDefaultCollapsedSections(),
            startedAt: Date.now(),
            failureMessage: null,
        };
        DebugSession.sessions.set(message.id, this);
    }

    public static getByMessageId(messageId: string): DebugSession | null {
        return this.sessions.get(messageId) || null;
    }

    public static getDefaultCollapsedSections(): Record<DebugSectionKey, boolean> {
        return { ...this.defaultCollapsedSections };
    }

    private static persistDefaultCollapsedSections(
        collapsedSections: Record<DebugSectionKey, boolean>
    ): void {
        this.defaultCollapsedSections = { ...collapsedSections };
    }

    public async toggleSection(section: DebugSectionKey): Promise<void> {
        await this.mutate(
            "Toggling debug section",
            `Toggled ${section} section`,
            (state) => {
                state.collapsedSections[section] = !state.collapsedSections[section];
                DebugSession.persistDefaultCollapsedSections(state.collapsedSections);
            }
        );
    }

    public async setAllSectionsCollapsed(collapsed: boolean): Promise<void> {
        await this.mutate(
            collapsed ? "Collapsing debug sections" : "Expanding debug sections",
            collapsed ? "Collapsed all debug sections" : "Expanded all debug sections",
            (state) => {
                for (const key of Object.keys(state.collapsedSections) as DebugSectionKey[]) {
                    state.collapsedSections[key] = collapsed;
                }
                DebugSession.persistDefaultCollapsedSections(state.collapsedSections);
            }
        );
    }

    public buildComponents() {
        return renderDebugTrace(this.state);
    }

    public async setClassifying(): Promise<void> {
        await this.mutate("Classifying request", "Classifying the turn");
    }

    public async setClassification(
        mode: "direct_answer" | "discord_grounded"
    ): Promise<void> {
        await this.mutate(
            "Classification ready",
            `Classification selected: ${mode}`,
            (state) => {
                state.classificationMode = mode;
            }
        );
    }

    public async setRequesterContext(requesterLabel: string, trigger: TurnTrigger): Promise<void> {
        await this.mutate(
            "Loading requester context",
            `Requester: ${requesterLabel} via ${trigger}`,
            (state) => {
                state.requesterLabel = requesterLabel;
                state.trigger = trigger;
            }
        );
    }

    public async setConversationContext(context: {
        threadId: string;
        kind: DebugTraceState["conversationContext"]["kind"];
        replyAnchorMessageId?: string | null;
        replyContext?: DebugTraceState["conversationContext"]["replyContext"];
    }): Promise<void> {
        await this.mutate(
            "Resolving conversation identity",
            `Conversation key: ${context.threadId}`,
            (state) => {
                state.checkpointThreadId = context.threadId;
                state.conversationContext = {
                    threadId: context.threadId,
                    kind: context.kind ?? null,
                    replyAnchorMessageId: context.replyAnchorMessageId || null,
                    replyContext: context.replyContext || null,
                };
            }
        );
    }

    public async setRuntimeMode(mode: RuntimeMode): Promise<void> {
        await this.mutate("Routing runtime", `Runtime mode: ${mode}`, (state) => {
            state.runtimeMode = mode;
        });
    }

    public async setCheckpointThread(threadId: string): Promise<void> {
        await this.mutate("Loading checkpoint", `Checkpoint thread: ${threadId}`, (state) => {
            state.checkpointThreadId = threadId;
        });
    }

    public async setPlanning(step: number): Promise<void> {
        await this.mutate("Planning next action", `Planning step ${step}`);
    }

    public async setWebStatus(status: WebStatus): Promise<void> {
        await this.mutate("Updating web policy", `Web mode: ${status}`, (state) => {
            state.webStatus = status;
        });
    }

    public async setToolRunning(toolName: string, details: string[] = []): Promise<void> {
        await this.mutate(`Running ${toolName}`, `Running ${toolName}`, undefined, details);
    }

    public async setToolResult(
        toolName: string,
        summary: string,
        itemCount?: number
    ): Promise<void> {
        await this.mutate(
            `Tool result: ${toolName}`,
            itemCount === undefined
                ? `${toolName}: ${summary}`
                : `${toolName}: ${summary} (${itemCount})`,
            (state) => {
                state.toolCallCount += 1;
                if (!state.selectedCapabilities.includes(toolName)) {
                    state.selectedCapabilities.push(toolName);
                }
            }
        );
    }

    public async setToolProgress(toolName: string, summary: string): Promise<void> {
        await this.mutate(`Tool progress: ${toolName}`, `${toolName}: ${summary}`);
    }

    public async setRetrievalSummary(summary: RetrievalSummary): Promise<void> {
        await this.mutate(
            "Updating retrieval summary",
            `Retrieval origin=${summary.sourceOrigin}; strong=${summary.strongResultCount}; weak=${summary.weakResultCount}`,
            (state) => {
                state.retrievalSummary = summary;
            }
        );
    }

    public async setGroundingSummary(
        summary: { messageEvidenceCount: number; liveEvidenceCount: number; sufficient: boolean },
        _decisionMode?: string,
        answerMode?: GroundedAnswerMode
    ): Promise<void> {
        await this.mutate(
            "Judging evidence",
            summary.sufficient
                ? `Evidence sufficient: messages ${summary.messageEvidenceCount}, live ${summary.liveEvidenceCount}`
                : `Evidence still weak: messages ${summary.messageEvidenceCount}, live ${summary.liveEvidenceCount}`,
            (state) => {
                state.groundingSummary = summary;
                state.groundedAnswerMode = answerMode || null;
            }
        );
    }

    public async setStopReason(reason: StopReason, detail?: string | null): Promise<void> {
        await this.mutate(
            "Evaluating stop condition",
            detail ? `Stop reason: ${reason} (${detail})` : `Stop reason: ${reason}`,
            (state) => {
                state.stopReason = reason;
                state.stopDetail = detail || null;
            }
        );
    }

    public async setConfidence(confidence: GroundedAnswerMode): Promise<void> {
        await this.mutate("Updating confidence", `Confidence: ${confidence}`, (state) => {
            state.groundedAnswerMode = confidence;
        });
    }

    public async setTraceEvent(label: string, detail: string): Promise<void> {
        await this.mutate(label, detail);
    }

    public async setContextPreview(preview: {
        recentChannelMessages: string[];
        evidencePreview: string[];
        recentTurns: string[];
    }): Promise<void> {
        await this.mutate(
            "Loading context preview",
            `Channel: ${preview.recentChannelMessages.length} msgs, Evidence: ${preview.evidencePreview.length}, Turns: ${preview.recentTurns.length}`,
            (state) => {
                state.contextPreview = preview;
            }
        );
    }

    public async setGenerating(): Promise<void> {
        await this.mutate("Generating answer", "Generating the final user-facing answer");
    }

    public async finishSuccess(summary = "Response completed"): Promise<void> {
        await this.mutate(
            "Completed",
            summary,
            (state) => {
                state.status = "completed";
                state.stage = "Completed";
            }
        );
    }

    public async finishError(error: unknown): Promise<void> {
        const message = normalizeError(error);
        await this.mutate(
            "Failed",
            `Error: ${message}`,
            (state) => {
                state.status = "failed";
                state.stage = "Failed";
                state.failureMessage = message;
            }
        );
    }

    private async mutate(
        stage: string,
        event: string,
        mutateState?: (state: DebugTraceState) => void,
        extraEvents: string[] = []
    ): Promise<void> {
        this.updateQueue = this.updateQueue
            .then(async () => {
                this.state.stage = stage;
                const normalizedExtraEvents = extraEvents
                    .map((item) => item.trim())
                    .filter(Boolean);
                this.state.recentEvents = [
                    event,
                    ...normalizedExtraEvents,
                    ...this.state.recentEvents,
                ].slice(0, MAX_EVENTS);

                this.state.timeline = [
                    {
                        label: stage,
                        detail: event,
                        tone: inferTone(event),
                        timestamp: Date.now(),
                    },
                    ...normalizedExtraEvents.map((detail) => ({
                        label: stage,
                        detail,
                        tone: inferTone(detail),
                        timestamp: Date.now(),
                    })),
                    ...this.state.timeline,
                ].slice(0, MAX_TIMELINE);

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

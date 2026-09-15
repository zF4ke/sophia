export interface SourceChange { messageId?: string; url?: string; kind: "edited" | "deleted" | "unavailable" }
export class ExecutionStopped extends Error {
    constructor(public readonly reason: "cancelled" | "limit_reached" | "stalled" | "uncertain_action" | "persistence_failed" | "context_full") {
        super(reason === "context_full" ? "Required context exceeds model capacity." : reason === "cancelled" ? "Execution cancelled." : reason === "stalled" ? "Repeated tool failures." : reason === "uncertain_action" ? "External action outcome is unknown." : reason === "persistence_failed" ? "Could not preserve task evidence." : "Explicit tool-call limit reached.");
    }
}

/** One execution scope survives continuation turns. Zero means no operator limit. */
export class ExecutionControl {
    requestTaskResume?: (taskId: string) => Promise<void>;
    taskResumeSelected = false;
    private static readonly active = new Set<ExecutionControl>();
    private readonly controller = new AbortController();
    private calls = 0;
    private failedRounds = 0;
    private readonly corrections: string[] = [];
    private acceptingSteering = true;
    private steeringWriter?: (text: string, contributorId?: string) => Promise<void>;
    private pendingSteering: Promise<void> = Promise.resolve();
    private steeringFailure = false;
    private boundTaskId?: string;
    private readonly sources = new Set<string>();
    private readonly replyMessages = new Set<string>();
    bindReplyMessage(messageId: string): void { this.replyMessages.add(messageId); }
    static bindTaskMessage(taskId: string, messageId: string): void {
        for (const execution of this.active) if (execution.boundTaskId === taskId) execution.bindReplyMessage(messageId);
    }
    static async steerForReply(actorId: string, channelId: string, messageId: string, text: string): Promise<"queued" | "not_found" | "ambiguous"> {
        const matches = [...this.active].filter(execution => execution.actorId === actorId && execution.channelId === channelId && execution.replyMessages.has(messageId) && !execution.signal.aborted && execution.acceptingSteering);
        if (!matches.length) return "not_found";
        if (matches.length !== 1) return "ambiguous";
        matches[0].steer(text);
        await matches[0].flushSteering();
        return "queued";
    }
    private readonly expectedDeletions = new Set<string>();
    watchSources(messageIds: string[]): void { for (const id of messageIds) this.sources.add(id); }
    expectSourceDeletion(messageIds: string[]): void { for (const id of messageIds) this.expectedDeletions.add(id); }
    isExpectedDeletion(messageId: string): boolean { return this.expectedDeletions.has(messageId); }
    private readonly changedSources = new Map<string, SourceChange>();
    get pendingSourceChanges(): SourceChange[] { return [...this.changedSources.values()]; }
    acknowledgeSourceChanges(changes: SourceChange[]): void {
        for (const change of changes) if (change.messageId && this.changedSources.get(change.messageId) === change) this.changedSources.delete(change.messageId);
    }
    clearSourceWatches(): void { this.sources.clear(); }
    static invalidateSource(messageId: string, detail?: Omit<SourceChange, "messageId">): void {
        for (const execution of this.active) {
            if (execution.expectedDeletions.has(messageId)) continue;
            if (execution.sources.has(messageId) && !execution.signal.aborted) {
                execution.changedSources.set(messageId, { messageId, kind: "unavailable", ...detail });
            }
        }
    }

    bindTask(taskId: string, writer: (text: string, contributorId?: string) => Promise<void>): void {
        if (this.boundTaskId && this.boundTaskId !== taskId) throw new Error("Execution task mismatch.");
        this.boundTaskId = taskId;
        this.steeringWriter = writer;
    }

    async flushSteering(): Promise<void> {
        await this.pendingSteering;
        if (this.steeringFailure) throw new ExecutionStopped("persistence_failed");
    }

    constructor(
        public readonly actorId: string,
        public readonly channelId: string | null,
        public readonly toolCallLimit = 0,
    ) {
        if (!Number.isSafeInteger(toolCallLimit) || toolCallLimit < 0) {
            throw new Error("toolCallLimit must be a non-negative integer; 0 disables the limit.");
        }
    }

    get signal(): AbortSignal { return this.controller.signal; }
    get toolCalls(): number { return this.calls; }
    get steeringRevision(): number { return this.corrections.length; }
    get steering(): readonly string[] { return [...this.corrections]; }
    get requiresOwnerApproval(): boolean { return this.corrections.some(text => text.startsWith("[Collaborator ")); }
    restoreSteering(corrections: string[]): void {
        if (this.corrections.length) throw new Error("Cannot overwrite live steering.");
        this.corrections.push(...corrections);
    }
    openSteering(): void { this.acceptingSteering = true; }
    closeSteering(): void { this.acceptingSteering = false; }

    steer(text: string, contributorId?: string): void {
        const correction = text.trim();
        if (!correction || correction.length > 4000) throw new Error("Steering must contain 1–4000 characters.");
        if (this.signal.aborted || !this.acceptingSteering) throw new Error("Execution no longer accepts steering.");
        this.corrections.push(correction);
        if (this.steeringWriter) {
            const write = this.steeringWriter;
            this.pendingSteering = this.pendingSteering.then(() => write(correction, contributorId)).catch(() => { this.steeringFailure = true; });
        }
    }

    static async steerAsCollaborator(actorId: string, channelId: string, text: string, taskId: string, authorize: () => Promise<boolean>): Promise<"queued" | "not_found"> {
        const execution = [...this.active].find(item => item.boundTaskId === taskId && item.channelId === channelId && !item.signal.aborted && item.acceptingSteering);
        if (!execution || !await authorize()) return "not_found";
        execution.steer(`[Collaborator ${actorId}] ${text}`.slice(0, 4000), actorId);
        await execution.flushSteering();
        return "queued";
    }

    static async steerForActor(actorId: string, channelId: string | null, text: string, taskId?: string): Promise<"queued" | "not_found" | "ambiguous"> {
        const matches = [...this.active].filter(execution => execution.actorId === actorId &&
            execution.channelId === channelId && (!taskId || execution.boundTaskId === taskId) && !execution.signal.aborted && execution.acceptingSteering);
        if (!matches.length) return "not_found";
        if (matches.length > 1) return "ambiguous";
        matches[0].steer(text);
        await matches[0].flushSteering();
        return "queued";
    }

    register(): () => void {
        ExecutionControl.active.add(this);
        return () => { ExecutionControl.active.delete(this); };
    }

    checkpoint(): void {
        if (this.steeringFailure) throw new ExecutionStopped("persistence_failed");
        if (this.signal.aborted) throw new ExecutionStopped("cancelled");
        if (this.toolCallLimit > 0 && this.calls >= this.toolCallLimit) {
            throw new ExecutionStopped("limit_reached");
        }
    }

    beforeTool(): void {
        this.checkpoint();
        this.calls += 1;
    }

    observeRound(allToolsFailed: boolean): void {
        this.failedRounds = allToolsFailed ? this.failedRounds + 1 : 0;
        if (this.failedRounds >= 3) throw new ExecutionStopped("stalled");
    }

    cancel(): void { this.controller.abort(); }

    static cancelTask(taskId: string, actorId: string, channelId: string | null): boolean {
        const execution = [...this.active].find(item => item.boundTaskId === taskId && item.actorId === actorId && item.channelId === channelId);
        if (!execution) return false;
        execution.cancel();
        return true;
    }

    static cancelForActor(actorId: string, channelId: string | null): number {
        let count = 0;
        for (const execution of this.active) {
            if (execution.actorId === actorId && execution.channelId === channelId) {
                execution.cancel();
                count += 1;
            }
        }
        return count;
    }
}

import type { RetrievalSummary, ToolArguments, ToolArgumentValue } from "@/runtime/contracts";

/**
 * Runtime-owned state for "the user explicitly asked for N messages" tasks.
 *
 * Activated when either:
 * - the user explicitly asked for a corpus larger than the default page size, or
 * - the model's `retrieve_messages` call passes a `limit` strictly greater than
 *   the configured default page size.
 *
 * Updated only from real retrieval results (NEVER from model-authored notes or
 * plans). Read by the finish guard and by the long-task synthesis fallback so
 * that an under-collected run cannot be presented to the user as a completed
 * analysis.
 */
export interface CorpusTaskState {
    /** Target size declared by the model's first qualifying retrieve_messages call. */
    requested: number;
    /** Runtime-trusted running count (accumulated unique message count). */
    collected: number;
    /** True if history OR semantic continuation cursor is still available. */
    continuationAvailable: boolean;
    /** True once history is fully drained. */
    historyExhausted: boolean;
    /** Cached target author from tool args (useful for fallback messages). */
    targetAuthorId?: string;
    /** Cached target channels from tool args. */
    targetChannelIds: string[];
    /** Unified strike counter across finish and raw-text rejections. */
    guardViolations: number;
    /** Saved latest retrieval summary so the runtime can force a continuation call. */
    lastRetrieval: RetrievalSummary | null;
    /**
     * Full snapshot of the tool args from the first qualifying retrieve_messages
     * call — reused as the base for forced continuations so that
     * mode/order/fromDate/beforeTimestamp/afterTimestamp/excludedMessageIds and
     * any other shape-defining fields are preserved.
     */
    originalArgs: ToolArguments;
}

/**
 * After this many guard rejections (summed across finish-time and raw-text
 * paths), the runtime forces the next step to be another retrieve_messages
 * call rather than relying on the model to self-correct.
 */
export const FORCE_CONTINUATION_AFTER_VIOLATIONS = 2;

/**
 * Create (lazily) or update the corpus task state from a successful
 * `retrieve_messages` result. Returns the updated state, or the previous
 * state unchanged if the call is not a corpus task.
 *
 * @param defaultPageSize The configured retrieval page size.
 * @param requestedCorpusSize Optional corpus size parsed from the user's
 *        request. When present and larger than a normal page, it activates the
 *        state machine even if the model uses a small exploratory `limit`.
 */
export function createOrUpdate(
    state: CorpusTaskState | undefined,
    retrieval: RetrievalSummary | null,
    toolArgs: ToolArguments,
    defaultPageSize: number,
    requestedCorpusSize?: number | null,
): CorpusTaskState | undefined {
    if (!retrieval) return state;

    const rawLimit = toolArgs.limit;
    const limit = typeof rawLimit === "number" && Number.isFinite(rawLimit) ? rawLimit : 0;
    const requestedFromUser =
        typeof requestedCorpusSize === "number" && Number.isFinite(requestedCorpusSize)
            ? requestedCorpusSize
            : 0;
    const activationTarget = Math.max(limit, requestedFromUser);

    // No existing state + neither the user's request nor the tool limit implies
    // a large corpus → not a corpus task.
    if (!state && activationTarget <= defaultPageSize) return state;

    const authorIdRaw = toolArgs.authorId ?? toolArgs.author_id;
    const targetAuthorId = typeof authorIdRaw === "string" ? authorIdRaw : state?.targetAuthorId;

    const channelIdsRaw = toolArgs.channelIds;
    const argChannelIds = Array.isArray(channelIdsRaw)
        ? (channelIdsRaw.filter((x): x is string => typeof x === "string"))
        : [];
    const targetChannelIds = argChannelIds.length > 0
        ? argChannelIds
        : (state?.targetChannelIds ?? retrieval.activeChannelIds ?? []);

    const requested = state?.requested ?? activationTarget;
    const collected = Math.max(state?.collected ?? 0, retrieval.accumulatedUniqueCount ?? 0);
    // Snapshot the full original tool args on first activation so forced
    // continuations preserve mode / order / fromDate / beforeTimestamp /
    // afterTimestamp / excludedMessageIds and any other shape fields.
    const originalArgs = state?.originalArgs ?? { ...toolArgs };

    return {
        requested,
        collected,
        continuationAvailable: Boolean(retrieval.continuationAvailable),
        historyExhausted: Boolean(retrieval.historyExhausted),
        targetAuthorId,
        targetChannelIds,
        guardViolations: state?.guardViolations ?? 0,
        lastRetrieval: retrieval,
        originalArgs,
    };
}

/**
 * True while the run has not collected the requested number of messages and
 * more history is still reachable. This is the single invariant: a completed
 * answer is only allowed once `collected >= requested` or `historyExhausted`.
 */
export function isIncomplete(state: CorpusTaskState | undefined): state is CorpusTaskState {
    if (!state) return false;
    if (state.historyExhausted) return false;
    if (state.collected >= state.requested) return false;
    return state.continuationAvailable;
}

/**
 * Finish / raw-text rejection predicate. Callers must also call
 * `recordViolation` when a rejection actually fires.
 */
export function shouldRejectFinish(state: CorpusTaskState | undefined): state is CorpusTaskState {
    return isIncomplete(state);
}

/**
 * Record a guard violation. Mutates state (request-scoped).
 */
export function recordViolation(state: CorpusTaskState): CorpusTaskState {
    state.guardViolations += 1;
    return state;
}

/**
 * True once the violation count has reached the threshold at which the
 * runtime should stop asking the model nicely and force a continuation.
 */
export function shouldForceContinuation(state: CorpusTaskState | undefined): state is CorpusTaskState {
    if (!state) return false;
    return isIncomplete(state) && state.guardViolations >= FORCE_CONTINUATION_AFTER_VIOLATIONS;
}

/**
 * Corrective message pushed back to the model after a rejected finish /
 * raw-text attempt.
 */
export function buildRejectionMessage(state: CorpusTaskState): string {
    return (
        `Rejected: the user asked for ${state.requested} messages and only ` +
        `${state.collected} have been collected so far. A continuation cursor is ` +
        `still available. Call retrieve_messages again with the saved cursor and ` +
        `keep paginating until you reach ${state.requested} messages or history is exhausted.`
    );
}

/**
 * Synthetic tool arguments for a forced continuation call — the runtime
 * executes `retrieve_messages` with these args directly, bypassing the
 * stalled model.
 */
export function buildForcedRetrieveArgs(state: CorpusTaskState): ToolArguments {
    // Start from the full original args so mode/order/fromDate/beforeTimestamp/
    // afterTimestamp/excludedMessageIds/etc. are preserved. Then overlay the
    // fresh cursor from the latest retrieval.
    const args: ToolArguments = { ...state.originalArgs };
    if (!args.query || typeof args.query !== "string") args.query = "continuation";
    // Drop any cursor captured on the first call — it is stale by definition.
    // The fresh cursor from the latest retrieval is reattached below.
    delete args.cursor;

    const last = state.lastRetrieval;
    if (last) {
        const cursor: { [key: string]: ToolArgumentValue } = {};
        if (last.historyCursorByChannel && Object.keys(last.historyCursorByChannel).length > 0) {
            const historyCursor: { [key: string]: ToolArgumentValue } = {};
            for (const [k, v] of Object.entries(last.historyCursorByChannel)) {
                if (typeof v === "string") historyCursor[k] = v;
            }
            if (Object.keys(historyCursor).length > 0) cursor.history = historyCursor;
        }
        if (last.semanticCursor) {
            cursor.semantic = last.semanticCursor as unknown as { [key: string]: ToolArgumentValue };
        }
        if (Object.keys(cursor).length > 0) args.cursor = cursor;
    }
    return args;
}

/**
 * Deterministic incomplete-progress string for long-task synthesis.
 *
 * One neutral bilingual form — no per-call language heuristics. Until a
 * proper shared language resolver exists in the runtime, this is the
 * honest, low-sprawl option.
 */
export function incompleteFallback(state: CorpusTaskState): string {
    return (
        `Tarefa não concluída — coletei ${state.collected} de ${state.requested} mensagens ` +
        `antes de a execução ser interrompida. / Task incomplete — collected ` +
        `${state.collected} of ${state.requested} messages before the run was interrupted.`
    );
}

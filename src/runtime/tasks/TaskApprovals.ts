import { ExecutionStopped } from "../ExecutionControl";
import { taskStore } from "./TaskStore";

const wrapped = new WeakMap<Function, { binding: string; original: Function; signal?: AbortSignal }>();

/** Persistence is an audit trail, never a reusable grant or a model-owned approval. */
export function durableApproval<TRequest extends { requesterId: string }, TResult>(
    context: { taskId?: string; actorId: string; channelId: string; guildId: string | null; signal?: AbortSignal },
    gate: ((request: TRequest, signal?: AbortSignal) => Promise<TResult>) | undefined,
): typeof gate {
    if (!gate || !context.taskId) return gate;
    const binding = JSON.stringify([context.taskId, context.actorId, context.channelId, context.guildId]);
    const previous = wrapped.get(gate);
    if (previous?.binding === binding && previous.signal === context.signal) return gate;
    const original = (previous?.original ?? gate) as NonNullable<typeof gate>;
    const taskId = context.taskId;
    const callback = async (request: TRequest): Promise<TResult> => {
        if (request.requesterId !== context.actorId) throw new Error("Approval requester mismatch.");
        let id: string;
        try { id = await taskStore.beginApproval({ ...context, taskId, request: structuredClone(request) }); }
        catch { throw new ExecutionStopped("persistence_failed"); }
        let result: TResult;
        try {
            if (context.signal?.aborted) throw new ExecutionStopped("cancelled");
            result = await original(structuredClone(request), context.signal);
            if (context.signal?.aborted) throw new ExecutionStopped("cancelled");
        }
        catch (error) {
            await taskStore.settleApproval(id, context.actorId, { error: error instanceof Error ? error.message : String(error) }, "failed").catch(() => {});
            throw error;
        }
        try { await taskStore.settleApproval(id, context.actorId, result); }
        catch { throw new ExecutionStopped("persistence_failed"); }
        return result;
    };
    wrapped.set(callback, { binding, original, signal: context.signal });
    return callback;
}

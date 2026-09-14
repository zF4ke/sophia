import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import type { ModelProfile } from "@/shared/appTypes";
import { ModelRequestQueue } from "./ModelRequestQueue";
import { SettingsService } from "@/app/SettingsService";

export interface UsageScope { taskId?: string; actorId?: string; requestId?: string; job?: string }
export interface ModelUsageRecord extends UsageScope {
    id: string; model: string; purpose: string; status: "returned" | "failed";
    promptTokens: number | null; completionTokens: number | null; durationMs: number;
    estimatedCostUsd: number | null; errorCode: string | null; createdAt: number;
}
type ReturnedUsage = { usage?: { prompt_tokens?: number; completion_tokens?: number; promptTokens?: number; completionTokens?: number } | null };
const nonnegative = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;

/** Accounting follows async work, including nested tools. It never sets an execution budget. */
export class ModelUsage {
    private static readonly context = new AsyncLocalStorage<UsageScope & { signal?: AbortSignal }>();
    private static readonly queue = new ModelRequestQueue(() => SettingsService.load().runtime.modelConcurrency);
    static bindExecution(signal: AbortSignal): void { const scope = this.context.getStore(); if (scope) scope.signal = signal; }
    static signal(): AbortSignal | undefined { return this.context.getStore()?.signal; }
    static sessionId(): string | undefined { const scope = this.context.getStore(); return scope?.taskId ?? scope?.requestId; }
    static scope<T>(scope: UsageScope, work: () => T): T { return this.context.run({ ...this.context.getStore(), ...scope }, work); }
    static async measure<T>(profile: Pick<ModelProfile, "chatModel" | "pricing">, purpose: string, work: () => Promise<T>): Promise<T> {
        let started = Date.now();
        let dispatched = false;
        let value: T | undefined;
        let failed = false;
        let failure: unknown;
        const { signal, ...scope } = this.context.getStore() ?? {};
        try { value = await this.queue.run(scope.actorId ?? scope.taskId ?? "maintenance", Boolean(scope.job), signal, () => {
            dispatched = true;
            started = Date.now();
            return work();
        }); } catch (error) { if (!dispatched) throw error; failed = true; failure = error; }
        const usage = (value as ReturnedUsage | undefined)?.usage;
        const promptTokens = nonnegative(usage?.promptTokens ?? usage?.prompt_tokens);
        const completionTokens = nonnegative(usage?.completionTokens ?? usage?.completion_tokens);
        const pricing = profile.pricing;
        const record: ModelUsageRecord = {
            ...scope, id: randomUUID(), model: profile.chatModel, purpose,
            status: failed ? "failed" : "returned", promptTokens, completionTokens,
            durationMs: Math.max(0, Date.now() - started), createdAt: Date.now(),
            estimatedCostUsd: promptTokens !== null && completionTokens !== null && pricing?.inputPerMillionUsd !== undefined && pricing.outputPerMillionUsd !== undefined
                ? (promptTokens * pricing.inputPerMillionUsd + completionTokens * pricing.outputPerMillionUsd) / 1_000_000 : null,
            errorCode: failed ? String((failure as { status?: unknown; code?: unknown } | null)?.status ?? (failure as { code?: unknown } | null)?.code ?? "provider_error").slice(0, 80) : null,
        };
        const { taskStore } = await import("@/runtime/tasks/TaskStore");
        await taskStore.recordModelUsage(record);
        if (failed) throw failure;
        return value as T;
    }
}

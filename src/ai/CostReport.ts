import type { Client } from "@libsql/client";

export type CostWindow = "day" | "week" | "month" | "all";
export interface CostTotals { attempts: number; failures: number; knownCostUsd: number; unpriced: number; inputTokens: number; outputTokens: number; missingTokens: number; background: number }
export interface CostReport { window: CostWindow; since: number; until: number; totals: CostTotals; models: Array<CostTotals & { model: string }> }
const aggregate = `COUNT(*) AS attempts, SUM(json_extract(record_json,'$.status')='failed') AS failures,
    SUM(json_extract(record_json,'$.estimatedCostUsd')) AS knownCostUsd,
    SUM(json_extract(record_json,'$.estimatedCostUsd') IS NULL) AS unpriced,
    SUM(json_extract(record_json,'$.promptTokens')) AS inputTokens, SUM(json_extract(record_json,'$.completionTokens')) AS outputTokens,
    SUM(json_extract(record_json,'$.promptTokens') IS NULL OR json_extract(record_json,'$.completionTokens') IS NULL) AS missingTokens,
    SUM(json_extract(record_json,'$.job') IS NOT NULL) AS background`;
const totals = (row: Record<string, unknown>): CostTotals => Object.fromEntries(["attempts", "failures", "knownCostUsd", "unpriced", "inputTokens", "outputTokens", "missingTokens", "background"].map(key => [key, Number(row[key] ?? 0)])) as unknown as CostTotals;

/** Aggregate stored attempt prices, not today's prices. No prompts or task content are selected. */
export async function readCostReport(client: Client, actorId: string | null, window: CostWindow, now = Date.now()): Promise<CostReport> {
    const days = { day: 1, week: 7, month: 30, all: null }[window];
    if (days === undefined || (actorId !== null && !actorId)) throw new Error("Invalid cost report scope.");
    const since = days === null ? 0 : now - days * 86_400_000;
    const where = "created_at>=? AND created_at<=?" + (actorId === null ? "" : " AND actor_id=?");
    const args = [since, now, ...actorId === null ? [] : [actorId]];
    const [summary, models] = await client.batch([
        { sql: `SELECT ${aggregate} FROM model_usage WHERE ${where}`, args },
        { sql: `SELECT json_extract(record_json,'$.model') AS model, ${aggregate} FROM model_usage WHERE ${where} GROUP BY model ORDER BY knownCostUsd DESC,attempts DESC,model LIMIT 8`, args },
    ], "read");
    return { window, since, until: now, totals: totals(summary.rows[0]), models: models.rows.map(row => ({ ...totals(row), model: String(row.model) })) };
}

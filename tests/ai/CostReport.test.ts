import { createClient } from "@libsql/client";
import { expect, it } from "vitest";
import { readCostReport } from "@/ai/CostReport";

it("keeps personal totals isolated and counts unpriced retries and background work honestly", async () => {
    const client = createClient({ url: ":memory:" });
    try {
        await client.execute("CREATE TABLE model_usage(actor_id TEXT,created_at INTEGER,record_json TEXT)");
        const now = 10 * 86_400_000;
        const insert = (actor: string | null, age: number, record: object) => client.execute({ sql: "INSERT INTO model_usage VALUES(?,?,?)", args: [actor, now - age, JSON.stringify({ model: "model", status: "returned", ...record })] });
        await insert("alice", 0, { estimatedCostUsd: 0.25, promptTokens: 100, completionTokens: 20 });
        await insert("alice", 1, { status: "failed", estimatedCostUsd: null, promptTokens: null, completionTokens: null });
        await insert("alice", 86_400_000, { estimatedCostUsd: 0, promptTokens: 1, completionTokens: 0 });
        await insert("alice", 86_400_001, { estimatedCostUsd: 3 });
        await insert("bob", 0, { estimatedCostUsd: 9, promptTokens: 3, completionTokens: 4 });
        await insert(null, 0, { model: "background", job: "index", estimatedCostUsd: null });
        await insert("alice", -1, { estimatedCostUsd: 100 });
        const own = await readCostReport(client, "alice", "day", now);
        expect(own.totals).toEqual({ attempts: 3, failures: 1, knownCostUsd: 0.25, unpriced: 1, inputTokens: 101, outputTokens: 20, missingTokens: 1, background: 0 });
        expect((await readCostReport(client, null, "day", now)).totals).toMatchObject({ attempts: 5, knownCostUsd: 9.25, unpriced: 2, background: 1 });
        expect((await readCostReport(client, "alice", "all", now)).totals.knownCostUsd).toBe(3.25);
        expect((await readCostReport(client, "nobody", "all", now)).totals.attempts).toBe(0);
    } finally { client.close(); }
});

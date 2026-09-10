import { OperationalStore } from "@/runtime/storage/OperationalStore";

const DIGEST_PREVIEW_ITEMS = 5;
const DIGEST_KEY_CHARS = 40;
const DIGEST_VALUE_CHARS = 80;
const DIGEST_PREVIEW_BUDGET = 440;

/**
 * Tiny always-injected memory hint for the system prompt (~50-500 chars).
 *
 * Injecting the full memory list every turn would bloat context, but injecting
 * nothing means the model never learns `memory_search`/`memory_remember` exist.
 * A short digest of counts plus the most recent keys keeps the store
 * discoverable at a fixed, tiny token cost.
 *
 * Visibility matches `memory_search`: guild-shared memories plus the acting
 * user's own user-scoped memories. Other users' memories are never surfaced.
 */
export async function buildMemoryDigest(
    guildId: string | null,
    actorId: string | null
): Promise<string> {
    if (!guildId) {
        return "Not available in DMs.";
    }
    try {
        await OperationalStore.initialize();
        const client = OperationalStore.getClient();
        const rows = (
            await client.execute({
                sql: `
                    SELECT key, value, user_id
                    FROM long_term_memories
                    WHERE (guild_id = :guildId OR guild_id IS NULL)
                      AND (user_id IS NULL OR user_id = :actorId)
                    ORDER BY updated_timestamp DESC
                    LIMIT 200
                `,
                args: { guildId, actorId: actorId ?? "" },
            })
        ).rows as Array<Record<string, unknown>>;

        if (!rows.length) {
            return "No long-term memories saved yet. Save durable facts with memory_remember; recall them with memory_search.";
        }

        const mine = rows.filter((row) => row.user_id != null).length;
        const shared = rows.length - mine;

        const previews: string[] = [];
        let used = 0;
        for (const row of rows.slice(0, DIGEST_PREVIEW_ITEMS)) {
            const key = String(row.key || "").slice(0, DIGEST_KEY_CHARS);
            const value = String(row.value || "").replace(/\s+/g, " ").trim().slice(0, DIGEST_VALUE_CHARS);
            const part = `${key}: ${value}`;
            if (previews.length > 0 && used + part.length > DIGEST_PREVIEW_BUDGET) {
                break;
            }
            previews.push(part);
            used += part.length + 3;
        }

        const remaining = rows.length - previews.length;
        const moreNote = remaining > 0
            ? ` (+${remaining} more. Call memory_search to recall them.)`
            : " (Call memory_search to recall them.)";
        return `Saved memories: ${shared} shared, ${mine} from this user. Recent: ${previews.join(" | ")}${moreNote}`;
    } catch {
        return "Memory status: unavailable.";
    }
}

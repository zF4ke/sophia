import { z } from "zod";
import { T } from "@/shared/discordTools";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { buildPrefixFtsQuery, normalizeForLexicalMatch, tokenizeQueryTerms } from "@/shared/textSearch";
import type { ToolDefinition } from "./types";

async function ensureTable() {
    await OperationalStore.initialize();
    const c = OperationalStore.getClient();
    await c.executeMultiple(`
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id TEXT PRIMARY KEY,
            guild_id TEXT,
            user_id TEXT,
            kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_timestamp INTEGER NOT NULL,
            updated_timestamp INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ltm_guild_user ON long_term_memories(guild_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_ltm_key ON long_term_memories(key);
    `);
}

function now() { return Date.now(); }
function uid() { return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`; }

const memoryRememberParams = {
    type: "object",
    properties: {
        key: { type: "string", description: "Short key/label for the memory (e.g. 'user preference', 'project decision'). Max 64 chars." },
        value: { type: "string", description: "The fact to remember verbatim. Keep concise (max 2000 chars)." },
        scope: { type: "string", description: "Scope: 'guild' (shared), 'user' (per-user), or 'channel'. Default 'guild'." },
    },
    required: ["key", "value"],
} as const;

export const memoryRememberTool: ToolDefinition = {
    name: T.memory_remember,
    catalog: { effect: "write", description: "Save a long-term memory (persistent across sessions) for the guild/user.", evidenceRole: "discovery_only" },
    schema: {
        description: "Persist a fact to long-term memory. Use when the user says 'lembra-te que', 'remember that', or when you learn a durable preference, decision, or context that should survive beyond this thread. Stored per guild and optionally per user.",
        parameters: memoryRememberParams,
    },
    capability: {
        description: "Store long-term memory.",
        inputSchema: z.object({ key: z.string().min(2).max(64), value: z.string().min(2).max(2000), scope: z.string().optional() }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["persists memory"],
        async run(context, args) {
            await ensureTable();
            const key = String(args.key || "").trim().slice(0, 64);
            const value = String(args.value || "").trim().slice(0, 2000);
            if (!key || !value) return { tool: T.memory_remember, summary: "Missing key/value.", data: null, errorMessage: "key and value required." };
            const scope = String(args.scope || "guild").toLowerCase();
            const c = OperationalStore.getClient();
            const guildId = scope === "user" ? context.guild?.id ?? null : context.guild?.id ?? null;
            const userId = scope === "user" ? context.actorId ?? null : null;
            // channel scope is stored with guild+channel hint in key
            const id = uid();
            const ts = now();
            await c.execute({
                sql: `INSERT INTO long_term_memories (id, guild_id, user_id, kind, key, value, created_timestamp, updated_timestamp) VALUES (:id, :guildId, :userId, :kind, :key, :value, :ts, :ts)`,
                args: { id, guildId, userId, kind: scope, key, value, ts },
            });
            return {
                tool: T.memory_remember,
                summary: `Saved memory "${key}" (${scope})`,
                data: { id, key, value, scope, guildId, userId },
            };
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "🧠", labelPt: "Memorizar" },
};

const memorySearchParams = {
    type: "object",
    properties: {
        query: { type: "string", description: "Search query to find relevant memories (keywords; prefix matching, diacritic-insensitive). Leave empty to list recent." },
        limit: { type: "number", description: "Max results (1-20, default 8)." },
    },
    required: [],
} as const;

export const memorySearchTool: ToolDefinition = {
    name: T.memory_search,
    catalog: { effect: "read", description: "Search long-term memories for the current guild (and user).", evidenceRole: "discovery_only" },
    schema: {
        description: "Recall persistent memories. Use when you need prior context, preferences, or decisions saved across sessions. Search by keyword (prefix matching, diacritic-insensitive) or list recent.",
        parameters: memorySearchParams,
    },
    capability: {
        description: "Search long-term memories.",
        inputSchema: z.object({ query: z.string().max(200).optional(), limit: z.number().min(1).max(20).optional() }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["returns matching memories"],
        async run(context, args) {
            await ensureTable();
            const c = OperationalStore.getClient();
            const rawQuery = String(args.query || "").trim();
            const limit = Math.max(1, Math.min(20, Number(args.limit) || 8));
            const guildId = context.guild?.id ?? null;
            const actorId = context.actorId ?? null;
            // Guild-shared memories are visible to everyone in the guild;
            // user-scoped memories only to their owner.
            const scopeSql = "(m.guild_id = :guildId OR m.guild_id IS NULL) AND (m.user_id IS NULL OR m.user_id = :actorId)";
            const terms = tokenizeQueryTerms(rawQuery);
            let rows: Record<string, unknown>[];
            if (!terms.length) {
                const r = await c.execute({
                    sql: `SELECT m.* FROM long_term_memories m WHERE ${scopeSql} ORDER BY m.updated_timestamp DESC LIMIT :limit`,
                    args: { guildId, actorId, limit },
                });
                rows = r.rows as Record<string, unknown>[];
            } else {
                const ftsQuery = buildPrefixFtsQuery(terms);
                const candidateLimit = Math.max(limit * 6, 30);
                const r = await c.execute({
                    sql: `
                        SELECT m.*
                        FROM long_term_memories_fts fts
                        INNER JOIN long_term_memories m ON m.rowid = fts.rowid
                        WHERE long_term_memories_fts MATCH :ftsQuery AND ${scopeSql}
                        ORDER BY fts.rank
                        LIMIT :candidateLimit
                    `,
                    args: { ftsQuery, guildId, actorId, candidateLimit },
                });
                rows = (r.rows as Record<string, unknown>[])
                    .map((row) => {
                        const key = String(row.key || "");
                        const value = String(row.value || "");
                        const haystack = normalizeForLexicalMatch(`${key} ${value}`);
                        const lexicalScore = terms.reduce(
                            (total, term) => total + (haystack.includes(term) ? 1 : 0),
                            0
                        );
                        const recencyScore = Number(row.updated_timestamp || 0) / 1_000_000_000_000;
                        return { row, totalScore: lexicalScore + recencyScore, updated: Number(row.updated_timestamp || 0) };
                    })
                    .sort((left, right) =>
                        right.totalScore - left.totalScore ||
                        right.updated - left.updated
                    )
                    .slice(0, limit)
                    .map((scored) => scored.row);
            }
            const memories = rows.map((r) => ({
                id: String(r.id),
                key: String(r.key),
                value: String(r.value),
                kind: String(r.kind),
                updatedAt: Number(r.updated_timestamp),
            }));
            if (!memories.length) return { tool: T.memory_search, summary: rawQuery ? `No memories for "${rawQuery}"` : "No memories yet.", data: { memories: [] } };
            const summary = memories.map((m) => `• ${m.key}: ${m.value.slice(0, 120)}`).join("\n");
            return { tool: T.memory_search, summary: `Found ${memories.length} memories:\n${summary}`, data: { memories } };
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "🔍", labelPt: "Memória" },
};

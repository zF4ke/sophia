# Memory Indexing

Sophia keeps a local cache of Discord messages in libSQL.

## Purpose

The cache exists to make Discord retrieval cheaper and more durable over time. It is a permanent runtime state for Discord search, not a separate memory-search product.

## Retrieval Strategy

1. Search local cached/indexed Discord messages.
2. If evidence is weak, discover likely live channels.
3. Fetch live Discord messages.
4. Ingest them into the local cache.
5. Retry retrieval on the enriched cache.

## What The Cache Contains

Current runtime storage is focused on:
- `messages`
- `message_chunks`
- `channels`
- `index_state`
- `channel_crawl_state`
- `tool_runs`
- `runtime_runs`
- `trace_events`
- `conversation_messages`
- `long_term_memories` (plus the `long_term_memories_fts` FTS5 mirror, kept in sync by triggers and rebuilt on boot when missing)

Ingestion flattens non-text message bodies into searchable plain text: embeds become `title — description — fields` lines, and Components V2 messages (artifact cards, bot panels) have their component tree walked (`src/memory/ingest/ComponentTextExtractor.ts`), extracting text displays, section bodies, button labels with link URLs, and select options. This is why artifact cards are findable through `retrieve_messages` even though their raw Discord `content` is empty.

The cache is used to support Discord retrieval and runtime continuity. It is not an autonomous long-term belief system yet.

## Long-Term Memory

`memory_remember` / `memory_search` store durable guild/user facts in `long_term_memories`. Search is FTS5-backed: queries are tokenized (diacritics stripped, `[a-z0-9]` only) and matched as prefix OR queries, then re-ranked by lexical overlap plus recency. Guild-shared rows are visible to the whole guild; `user`-scoped rows only to their owner. A compact digest (counts + recent keys, capped ~500 chars) is injected into the system prompt every turn via `src/runtime/memoryDigest.ts` so the model discovers the store without a full-context dump.

## How Retrieval Uses It

- `retrieve_messages` searches the local cache first and escalates automatically when needed.
- `random_channel_message` samples one or more random already-ingested messages from a channel (pass `count` up to 25 for batched sampling), so users can ask for "N random messages from this chat" without hitting Discord live APIs and without repeated identical calls.
- `resolve_member_identity` resolves exact member and bot ids first, then falls back through live search and same-guild historical authors.
- `resolve_channel_targets` and `list_guild_structure` expose current-guild channels, categories, and cached-only remembered entries.
- `get_member_profile` fetches live identity and profile metadata.
- `list_members` fetches live guild membership data.
- `get_guild_context` fetches live guild metadata.
- `get_role_info` fetches live role details.
- `list_threads` and `read_thread_messages` provide thread discovery and reading.


## Commands

- `/index` manages backfill and repair
- `/index status` shows local retrieval memory, guild completeness, and runtime storage status, including the current DB size on disk
- `/nth` reads indexed historical messages

## Background crawl lifecycle

- Startup performs a bounded newest-first sweep. It does not enqueue full-history work.
- Legacy unbounded `startup_refresh` rows are retired before the worker begins.
- A demand-driven job left in `running` by a restart returns to `queued` on the next startup.
- A failed job moves to `paused` instead of retrying forever. Explicitly enqueueing that channel retries it.

## Reset Semantics

Runtime storage is treated as disposable local state.

If the local schema is stale or incompatible, the runtime should reset and rebuild its operational storage rather than surfacing raw SQL failures in user-facing replies.

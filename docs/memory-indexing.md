# Memory And Indexing

Sophia stores Discord messages locally so retrieval quality does not depend on Discord search.

## Write Path

- `src/discord/events/message/messageCreate.event.ts` ingests eligible new messages.
- `src/memory/ingest/` normalizes messages and filters out junk.
- `src/memory/index/` splits messages into chunks.
- `src/memory/repositories/MemoryWriteRepository.ts` writes messages, chunks, embeddings, and index state.

## Read Path

- `src/memory/search/MemorySearchService.ts` performs lexical plus embedding retrieval.
- `src/memory/repositories/MemoryReadRepository.ts` handles thread reads, summaries, and historical lookup.
- `src/memory/DiscordMemoryService.ts` is the small facade used by the rest of the app.
- Live fallback crawl uses the official Discord API to fetch channel messages, ingests them through the same memory service, and then reruns local retrieval.
- Channel crawl state is stored so the agent can distinguish indexed channels from readable channels that have not been crawled yet.

## Reuse And Caching

- Successful and partially useful grounded runs are stored as reusable grounded contexts in the same SQLite database.
- Reuse is guild-wide by default, but same-channel matches are preferred over other channels in the guild.
- Cached grounded contexts store evidence text, internal citations, tool runs, sufficiency state, TTL expiry, and the guild response ordinal they were created on.
- Deterministic or bounded-expensive tool results are cached separately from `tool_runs`.
- `crawl_channel_messages` is not cached as a reusable result blob; its side effect is the message ingestion itself.
- Cache invalidation uses both TTL and a short per-guild "responses ago" limit in v1.

## Tool Cache Policy

- Short TTL: `get_guild_context`, `get_member_profile`, `list_members` with a 10-minute cap
- Longer TTL: `search_messages`, `list_relevant_channels`, `read_message_thread`, `read_channel_summary` with a 20-minute cap
- Both caches also expire after roughly 6 later Sophia responses in the same guild.
- When a cache hit exists, the runtime returns the stored `DiscordToolResult` instead of calling the underlying tool again.

## Operational Commands

- `/index status`
- `/index backfill_channel`
- `/index backfill_category`
- `/index repair`
- `/index clear`

If retrieval ranking changes, update the tests and this document in the same change.

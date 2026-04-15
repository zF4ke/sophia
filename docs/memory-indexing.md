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

The cache is used to support Discord retrieval and runtime continuity. It is not an autonomous long-term belief system yet.

## How Retrieval Uses It

- `retrieve_messages` searches the local cache first and escalates automatically when needed.
- `resolve_member_identity` resolves exact member and bot ids first, then falls back through live search and same-guild historical authors.
- `resolve_channel_targets` and `list_guild_structure` expose current-guild channels, categories, and cached-only remembered entries.
- `get_member_profile` fetches live identity and profile metadata.
- `list_members` fetches live guild membership data.
- `get_guild_context` fetches live guild metadata.
- `get_role_info` fetches live role details.
- `list_threads` and `read_thread_messages` provide thread discovery and reading.

`/find` is a specialized retrieval workflow that uses the same shared retrieval primitives.

## Commands

- `/index` manages backfill and repair
- `/index status` shows local retrieval memory, guild completeness, and runtime storage status
- `/nth` reads indexed historical messages

## Reset Semantics

Runtime storage is treated as disposable local state.

If the local schema is stale or incompatible, the runtime should reset and rebuild its operational storage rather than surfacing raw SQL failures in user-facing replies.

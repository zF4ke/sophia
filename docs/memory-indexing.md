# Memory Indexing

Sophia keeps a local cache of Discord messages in libSQL.

## Purpose

Portable presentation preferences belong to the authenticated Discord user across enabled locations. The allowed keys are `response_length`, `tone` and `language`, with constrained values. Dreaming may retain an explicit durable preference; ordinary facts remain scoped. Preference recall omits original source links, while storage retains provenance for deletion. Corrections require revisions, and forgetting suppresses automatic recreation across locations. A current user request overrides a saved preference.

The operational cache supports Discord retrieval. Durable knowledge is separate from that disposable index.

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
- Legacy `long_term_memories` are copied into the durable knowledge store before an index rebuild.

Ingestion flattens non-text message bodies into searchable plain text: embeds become `title — description — fields` lines, and Components V2 messages (artifact cards, bot panels) have their component tree walked (`src/memory/ingest/ComponentTextExtractor.ts`), extracting text displays, section bodies, button labels with link URLs, and select options. This is why artifact cards are findable through `retrieve_messages` even though their raw Discord `content` is empty.

The cache supports retrieval; it does not own Sophia's remembered facts, task files, cards or saved procedures.

## Long-Term Memory

`KnowledgeStore` owns `storage/knowledge.sqlite`, one persistent Sophia identity, and a single FTS5 memory collection. `memory_remember` defaults to the current channel. Records retain owner, source location, scope, revision and sources. The runtime digest lists eligible labels rather than dumping stored values into the system prompt.

Channel memories stay in their source channel. Guild memories are explicitly shared within that guild. User memories require the owner and are available in that owner's DMs or the original source location. A common identity does not make private facts public. Legacy guild memories retain guild scope; records with ambiguous audience metadata are quarantined and excluded from retrieval.

`memory_update` and `memory_forget` require the authenticated owner and current revision. Forgetting clears the value and leaves a tombstone. An exact forgotten key cannot be automatically recreated. Dreaming receives existing facts and suppressed labels to avoid paraphrasing a forgotten fact. Semantic erasure across unrelated source material is not a deterministic guarantee.

Discord deletion events invalidate source URLs, remove indexed messages/chunks, tombstone memories derived from those URLs, and cancel pending or in-flight dreams from them. A stale crawl cannot reinsert a deleted message. Deleted-source markers survive index resets. Historical task evidence and already-delivered summaries still need separate source-aware invalidation; audit records are not currently erased by these hooks.

## Dreaming

After successful delivery, eligible conversations enter a durable queue. Ephemeral responses are excluded. During idle time `DreamingService` processes one queued conversation with current actor, location and channel permission checks before and after the model call. It loads `memory/dreaming`, has no tools, and can return up to five candidate facts. The service fixes their provenance and channel audience; the model cannot broaden access.

Each source turn is queued once. Failed jobs retry up to three attempts and remain inspectable after failure. This bounds retries for one job rather than imposing a foreground task budget. Existing and forgotten keys are not overwritten by dreaming. Automatic contradiction reconciliation is not implemented yet.

`memory.dreamingEnabled` and `memory.dreamIntervalMs` control the service. The settings panel can pause automatic memory. Memory, dream jobs and identity survive index resets.

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

Only operational message indexes and runtime diagnostics are reconstructible. Index reset clears retrieval tables transactionally after migrating legacy durable records. Tasks, knowledge, skills, cards, schedules, settings and access rules survive it. Durable migrations create backups before changing data; a failed backup blocks migration. See [cleanup and migration](cleanup-migration.md).

## Edited source revisions

`messageUpdate` refreshes enabled-location messages, fetching partial events first. Ingestion compares observed content and edit timestamps under the per-message lock. Changed messages invalidate the previous source revision, dependent memories and dreams, task evidence, working notes and derived files. Active readers pause. Older observed edit timestamps and already-invalidated revisions cannot overwrite newer indexed content. A partial event pauses active readers before fetching; failed refreshes are logged and await a later successful fetch.

Edited message links retain the Discord URL and add a `sophia-revision` fragment containing the edit timestamp and a content digest. The fragment identifies the observed text; Discord still opens the same message. Task provenance retains that exact source URL through context reuse, cards and dreaming. Deleting a message invalidates all its revisions. A later real edit, including a reversion to earlier text, can supply fresh evidence. Source-invalidated requests cannot contribute old summaries or late notes/files. Existing published messages are not retracted automatically.

## Local-first research

Ordinary research does not require a user-issued indexing command. Retrieval uses persisted messages, returns cursors and partial-index metadata, and queues missing older history when it reaches the indexed boundary. A queued crawl is not a completed crawl. The runtime prompt distinguishes the age of a message from proof that indexing is stale, so quiet channels do not trigger unnecessary refreshes merely because their newest message is old.

# Cleanup And Migration Notes

## Offline state transfer

Stop Sophia after completing v5 startup migration. `npm run state -- backup <new-directory>` acquires the same instance guard as the bot and snapshots settings, security directories, tasks, identity/memory and products. SQLite snapshots include committed WAL data. A manifest hashes each file. Credentials, operational indexes, logs and earlier migration backups are excluded.

`npm run state -- restore <archive-directory> <empty-storage-directory>` validates the manifest and checksums before restoring into an empty destination. It does not replace an existing deployment. Settings retain access rules while the operational index path moves under the restored directory. Start with `SOPHIA_STORAGE_ROOT` pointing there. Normal startup pauses interrupted tasks and revalidates schedules; restoration never replays actions. Keep the matching model profiles with the application and supply credentials separately. Failed transfers leave partial output for inspection, without a valid completed archive manifest or an overwritten deployment.

Legacy memories without source provenance enter quarantine, including old guild-wide facts. Operators review them through `/memories quarantine:true` and adopt exact revisions into private memory. Adoption does not restore an unknown shared audience.

V5 preserves settings, access rules and durable databases during `npm run clean`; that command removes only logs and test cache directories. Index reset first runs legacy migration. `knowledge.sqlite` owns identity and memory, `tasks.sqlite` owns work and receipts, and `products.sqlite` owns cards and saved procedures. These files remain outside disposable runtime folders. Ambiguous legacy memories remain quarantined rather than gaining broader visibility.

This codebase intentionally removed older prompt-harness and cache-heavy paths that conflicted with the current runtime.

## Removed Or Replaced

- Large inline prompt blobs were replaced by `resources/prompts/`.
- The old Gemini-only AI facade was replaced by the current model gateway.
- The LangGraph StateGraph runtime was replaced by a simple while-loop with native function calling.
- Separate model calls (`plan_turn`, `select_next_step`, `judge_evidence`, `synthesize_answer`) were replaced by one unified system prompt (`runtime/agent_loop`) and the `finish` tool.
- The `CheckpointStore` and LangGraph checkpoint persistence were removed. Conversation state is now managed through `DiscordMemoryService`.
- `DebugStateStore` was replaced by `SettingsService` for persistent bot settings.
- The debug section collapse/expand system and budget truncation logic were removed in favor of a clean two-container trace layout.
- Context stuffing and fake web-search behavior were removed from the active runtime.
- The monolithic security and UI services were split into smaller modules.
- Legacy novelty commands and dead helper modules were removed.
- Refusal-first grounding was replaced by best-effort conversational recovery.
- Removed dependencies: `@langchain/core`, `@langchain/langgraph`, `@langchain/langgraph-checkpoint-sqlite`, `dedent`.

## Current Direction

Sophia is conversational first.

Local storage separates disposable Discord retrieval state from durable tasks, knowledge, skills, schedules and cards.

When evidence is weak, the runtime should keep the conversation moving with the best grounded interpretation it can produce, then ask a targeted follow-up or continue retrieval.

Write and destructive actions follow the authenticated requester's action tier and Ask/Auto mode. Ask-mode destructive actions retain their confirmation step. Explicit destructive Auto grants may skip the prompt; protected targets remain blocked.

## V5 execution settings

`runtime.maxToolCalls` and `runtime.longTask.maxToolCalls` are removed on load. Their values are not imported as v5 limits. `runtime.toolCallLimit` defaults to 0, meaning no call cap. A positive value is an explicit limit shared across continuation turns. Other retrieval and context settings are preserved.

Settings writes use a temporary file and rename. Invalid settings now produce a load error instead of replacing the configuration with defaults. Repair the existing file before restarting; the invalid file is preserved for inspection.
Settings migration saves the original bytes under `storage/backups`, keyed by content hash. Durable database migrations and legacy imports create a consistent SQLite snapshot with `VACUUM INTO` before changing schemas or resetting the old source. Each snapshot has a source-path manifest and is retained across ordinary cleanup. A failed backup blocks migration. Backups can contain private data and should follow the same access and retention policy as the originals.

The obsolete inline-crawl batch setting is removed. `DiscordHistoryReader` owns page ingestion and contiguous-history checkpoints for foreground retrieval, refresh and deep backfill. The extra tool-level crawl and deferred ingestion queue are removed.

## V5 access migration

Task ownership and outcomes now persist in `storage/tasks.sqlite`. Back up this file with its SQLite sidecar files while the bot is stopped. Runtime/index resets leave it intact. A restart marks previously running records paused, without retrying their actions. The ledger tracks new runs; old runtime traces are not backfilled as task records.

An empty `guildAllowlist` now disables all servers. Existing explicit guild IDs are preserved, and missing `access` settings become empty grants with DMs disabled. Sophia does not infer a server or user grant from cached history. Existing operators can run `/access enable_here:true` in a disabled server, then grant users or roles through `/access`. Register the updated slash-command schema before using the new options. See [Access policy](access-policy.md) for examples and remaining privacy work.

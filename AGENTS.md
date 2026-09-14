# Sophia Agent Handbook

Sophia is a Discord assistant with one conversation runtime, one retrieval pipeline, and one local runtime state store.

This file is the engineering handoff for the current runtime.

## Canonical Docs

- `docs/architecture.md`
- `docs/agent-loop.md`
- `docs/memory-indexing.md`
- `docs/prompt-catalog.md`
- `docs/commands-and-admin.md`
- `docs/access-policy.md`
- `docs/task-lifecycle.md`
- `docs/cleanup-migration.md`
- `docs/testing.md`
- `docs/ui.md`
- `docs/how-sophia-works.md`
- `docs/feature-user-stories.md`
- `docs/workspace-and-media.md`
- `docs/skills.md`
- `docs/web-research.md`
- `docs/scheduling.md`

## Runtime Model

Sophia is conversational first.

The runtime uses a while-loop with native function calling. One unified system prompt (`runtime/agent_loop`) gives the model the question, context, and available tools. The model calls tools iteratively and calls `finish` when it has an answer.

The active flow is:
1. Normalize the incoming turn.
2. Resolve the canonical conversation key.
3. Load memory: recent turns, channel context, and prior evidence from persisted tool runs.
4. Build the unified system prompt.
5. Enter while-loop: model calls tools via `generateWithTools`, runtime executes and feeds results back.
6. Model calls `finish` with the answer, or runtime produces a conversational fallback.
7. Persist runtime and trace data locally.

The runtime keeps only narrow guardrails:
- capability validation
- repeated-call protection
- no default task call or duration cap; optional explicit execution limit and cooperative cancellation
- context overflow pruning (Tier-1 truncation + Tier-2 compaction)
- doom-loop detection (identical tool calls → nudge → force finish)
- progress-required tracking (long tasks only)
- refusal prevention for ordinary conversation

Intent policy:
- The active runtime is model-led inside `runtime/agent_loop`.
- Do not reintroduce a separate deterministic intent router or argument-enrichment layer unless there is a clear, measured need.

## Stable Tool Contract

`verify_action` checks exact postconditions for an unknown receipt in an owned paused task. Supported observations may settle the receipt without retrying or resuming; partial/unsupported results stay unknown. `/tasks verify:true` exposes the same verifier. Permission errors are not absence evidence.

`skill_evaluate` records a content-bound review against a completed owned task. Dream-created procedures require a passing evaluation before `skill_save` can mark matching content ready. Evaluation does not execute or approve a procedure.

Research collections use `corpus_create`, `corpus_collect`, and `corpus_read`. They own immutable filters, exact retrieval cursors, unique messages and coverage in tasks.sqlite. Collection and reading recheck the requester's source-channel permissions. Revisions protect concurrent pages; deleted message tombstones prevent stale ingestion. Page exports are task files, never a claim of complete Discord coverage.

Source-derived task files preserve message/channel provenance across transformations. Source deletion removes these files and saved tool text; stale writes cannot restore them. `ToolExecutor` checks message-evidence source access before model delivery, and persisted tool evidence is revalidated before replay/export. Summaries and notes still need broader provenance tracking; do not treat these checks as a complete disclosure model.

These capability ids are prompt- and runtime-stable:

- `retrieve_messages`
- `search_messages`
- `random_channel_message`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`
- `clear_messages`
- `delete_messages`
- `create_channel`
- `create_category`
- `measure_text_length`
- `evaluate_math`
- `get_role_info`
- `list_roles`
- `delete_channel`
- `delete_role`
- `create_thread`
- `move_channel`
- `move_category`
- `manage_member_roles`
- `list_threads`
- `read_thread_messages`
- `send_message`
- `edit_message`
- `create_role`
- `edit_channel`
- `edit_role`
- `start_long_task` prepares a larger evidence context for sustained research. It does not raise execution limits or require estimates. Tasks have no default call or duration cap. An optional runtime.toolCallLimit applies to the whole execution, including continuation turns; 0 disables it.
- `note_add`
- `note_list`
- `note_clear`
- `goal_open`
- `goal_update`
- `goal_done`
- `plan_update`
- `web_search`
- `fetch_url`
- `memory_search`
- `memory_remember`
- `memory_update`
- `memory_forget`
- `workflow_create`
- `workflow_list`
- `workflow_run`
- `workflow_delete`
- `artifact_send`
- `artifact_edit`
- `artifact_read` inspects owned card state and historical revisions; use its current revision when editing.
- `task_forget` deletes an owned inactive task's working data; unresolved actions must be verified first.
- `tool_search`
- `create_poll`
- `get_poll_results`
- `index_channel`
- `sandbox_run`
- `sandbox_import`
- `sandbox_publish`
- `sandbox_inspect`
- `sandbox_transcribe`
- `inspect_runtime`
- `skill_search`
- `skill_load`
- `skill_save`
- `skill_delete`
- `source_read`
- `schedule_create`
- `schedule_update`
- `schedule_list`
- `schedule_cancel`

Tool schemas are defined in `src/runtime/toolSchemas.ts`.

The stable code catalog is:

- `src/shared/discordTools.ts`

If a capability id changes, update:

- `src/shared/discordTools.ts`
- `src/runtime/toolSchemas.ts`
- `src/capabilities/CapabilityRegistry.ts`
- `resources/prompts/runtime/agent_loop.md`
- `resources/prompts/runtime/compaction.md`
- `resources/prompts/media/transcription.md`
- the affected tests

Mutating capability output contract (required for all current and future `write`/`destructive` tools):
- Return identifier-rich `data` fields that are directly reusable in user replies (for Discord resources include both raw IDs and mention-ready fields, e.g. `channelId` and `channelMention: <#id>`).
- Include the same concrete identifier in `summary` (not only display names), so the model can produce accurate post-action confirmations.
- Prefer stable identifiers over names when they differ in reliability.

## Prompt Ownership

Runtime prompts live under `resources/prompts/` and stay external to code:

- `resources/prompts/runtime/agent_loop.md`
- `resources/prompts/memory/dreaming.md`

Global voice is selectable via `/settings`: `balanced`, `casual`, or `formal`. It changes presentation, never identity or authority. Legacy personality settings migrate to voice and are removed. Retrieved context belongs in source messages, never interpolated into system instructions. Prompt parameters are replaced once, so inserted source text cannot expand another template variable.

The stable prompt catalog is:

- `src/shared/promptCatalog.ts`

If a prompt contract changes, update:

- the prompt file
- `src/runtime/contracts.ts`
- `src/runtime/Runtime.ts`
- `docs/prompt-catalog.md`
- related tests

## Models And Config

`ModelRequestQueue` limits simultaneous provider calls via `runtime.modelConcurrency`, one per actor, with foreground priority over queued background work. It never caps total task work. `ModelUsage.bindExecution` carries cooperative cancellation into the waiting queue; cancelled queued calls do not count as provider attempts.

Editable runtime config lives in:

- `resources/models/model-profiles.json`
- `storage/settings.json` (managed by `SettingsService`)
- `.env` (credentials: `DISCORD_TOKEN`, `OPENROUTER_API_KEY`, `OPENCODE_API_KEY`; bootstrap/storage options are documented in `.env.example`)
- `src/app/AppConfig.ts`

All runtime tuning knobs (tool calls, retrieval limits, etc.) live in `SettingsService` defaults and `storage/settings.json`. Do not use environment variables for runtime config.

Model profiles support OpenRouter, OpenCode Zen and explicit compatible endpoints, including local servers. Do not hardcode models in source. `api` explicitly selects chat-completions or Responses; credential names must not select a protocol. Zen receives the Sophia task ID as session metadata and an honest Sophia client identity. Reasoning effort is profile-owned. External profiles bypass OpenRouter routing and fallback; OpenRouter fallback must never send an external profile's ID to OpenRouter. Embeddings still route through OpenRouter.

Provider attempts are stored in `model_usage`. `/costs` reports the actor's retained usage; `/settings` → Custos is operator-only and reports installation totals. Unknown prices or tokens remain unknown. Accounting never imposes a spending cap.

The Starlight site in `website/` replaces the legacy landing pages. User guides are authored under `website/src/content/docs/use/`, setup guides under `setup/`, and canonical engineering pages under `docs/`. Build scripts generate capability/command references and copies of engineering docs. Run `npm run docs:build` and `npm run docs:check`; never edit generated copies. `scripts/docs-reference.ts` fails if a capability lacks a user-guide mapping.

## Memory And Storage

`StateArchive` and `npm run state` provide offline durable-state transfer under the instance guard. Restore validates hashes and uses an empty destination; it never replaces a running or populated deployment. Credentials, operational indexes and old backups are excluded. Startup recovery remains responsible for pausing interrupted work.

Durable knowledge and identity live in `storage/knowledge.sqlite`; cards and saved procedures live in `storage/products.sqlite`. `KnowledgeStore` and `ProductStore` copy legacy records before the operational database can be rebuilt. Missing legacy memory audiences are quarantined. Dreaming processes delivered, non-ephemeral conversations during idle time with current access checks and no tool execution. See `docs/memory-indexing.md` for disclosure and deletion boundaries.

Generated code and media decoders use `ContainerSandbox`. Do not execute model-written code in the host process. Task files remain owned by the authenticated actor and location. `npm run clean` preserves settings, access rules and all databases.

Durable task ownership and outcomes live in `storage/tasks.sqlite`, managed by `TaskStore`. Keep this ledger outside disposable runtime/index resets. Startup pauses records interrupted by a previous process session; it must never automatically replay external actions. Continuation task IDs are verified against the actor, location, session and active status. See `docs/task-lifecycle.md` for current recovery boundaries.

Task-bound mutations record intent before dispatch and save returned receipts in `task_actions`. Unknown outcomes pause execution; never assume an interrupted action failed or retry it automatically. `/tasks task_id` exports the owner's receipts. A succeeded receipt records the handler result, not independent verification of every postcondition.

`TaskWorkspace` stores plans, notes and goals in the same durable database. Pass the trusted `taskId` in capability context. Working-state reads and updates stay within that task, including continuation and `note_list` with its legacy history flag. Never use channel-wide goals to drive another task's continuation. `/tasks task_id:<id>` exports the owner's saved state without resuming actions.

Local libSQL storage is the source of truth for Discord retrieval state and runtime traces.

`DiscordHistoryReader` serializes history-page fetching, author enrichment, ingestion and contiguous-history checkpoint updates per channel. Foreground retrieval, startup refresh and deep backfill use this owner. Targeted samples never advance the global history boundary. Do not add tool-local fetch-and-ingest loops or deferred ingestion queues.

`ModelUsage` records each actual provider attempt, including retries and nested/background model work, in the durable task store. Missing tokens/cost stay null; configured-price estimates are not billing receipts. Async scope carries task ownership. Accounting never imposes a spending cap.

Task evidence lives in `task_tool_runs` in the durable task database. Runtime continuation loads only its verified task's records; channel-wide operational tool history is diagnostic. Preserve full output and cursor data independently of prompt context limits. If a result cannot be saved, pause without repeating the tool. Do not turn this execution log into a tool-result cache.

- New messages are ingested from `src/discord/events/message/messageCreate.event.ts`
- **Startup sweep**: `DiscordBackfillCrawler.sweepAllGuildChannels()` runs a *bounded* recency pass per channel on `clientReady` (cap: `runtime.startupSweepMaxMessages`, default 1000) — closes offline gaps without full-history crawls
- **Edge prefetch**: when `retrieve_messages` pagination touches the indexed boundary, the channel is auto-enqueued for deep backfill (`runtime.edgePrefetch`)
- **Agent self-service indexing**: the `index_channel` tool lets the model refresh (newest sweep) or deep-backfill (queued) channels on its own judgment, guided by the `Index Freshness` block in the system prompt. Deep backfills are agent-gated only — boot never walks full history
- Backfill and repair flows remain exposed through `src/discord/commands/system/index.command.ts` (manual override only)
- Operational storage is local and disposable
- Bot settings are persisted in `storage/settings.json`

Do not reintroduce reusable grounded-context caches or tool-result caches into the active runtime path.

## Access admission

Role membership changes and channel moves use the destructive/access-sensitive tier. `ToolExecutor` enforces protected channel IDs independently of the conversational approval UI, including a fresh check at dispatch. Role mutation outputs include actual changed IDs and mention fields.

Grants support optional ISO expiry and the `/access` grant browser. `ToolExecutor` serializes mutations per guild, then rechecks authorization and steering before dispatch. Publishing tools declare `publicationTarget` so source audiences are checked against the actual destination, even during a private conversation. New publication tools must declare this metadata. Unknown outcomes are never automatically replayed.

`/memories` inspects eligible facts. Operators can inspect quarantined legacy memories and adopt an exact revision as owner-private memory. Legacy records without source provenance do not enter recall automatically. `/tasks forget:true` removes owned inactive resolved work; separate remembered facts, approved skills and operator backups are retained independently.

Cards persist authenticated creator IDs. `artifact_edit` requires the creator or an operator; legacy cards without owner metadata are operator-editable only. Card scripts run inside Docker, with current authorization and normal approvals for proposed external sends.

`AccessPolicy` binds grants to authenticated Discord user IDs. Empty `guildAllowlist` enables nothing; DMs require explicit availability. `/access` configuration remains operator-only and works in disabled locations. All conversation adapters supply an authorization callback, and runtime/tool execution rechecks it. Reads run freely within a grant; changes follow Ask or Auto-approve within the granted tier. `ToolExecutor` requires caller-owned approval for Ask decisions; model arguments cannot provide approval. See `docs/access-policy.md` for migration and current boundaries.

## Commands

`/skills` lists procedures and drafts. Operators can review quarantined legacy records and explicitly adopt a supplied ID/revision as a private channel draft. Adoption is a control-plane action; ordinary model discovery never exposes quarantine records.

Core supported commands and entrypoints:

`/talk task_id:<id> message:...` explicitly resumes an owned paused/failed/cancelled task in its original location. Current authorization and new approval decisions apply. Unresolved actions block resume. Approval records and steering events are durable; old approval controls never replay after restart.

- `/talk`
- `/steer instruction:... task_id:...` redirects the authenticated user's active execution in the current channel; task_id is optional when unique. Text replies to that task's original message or progress message also steer it while active.
- `/tasks` lists owned task outcomes, exports files/evidence/usage, and records explicit owner verification of uncertain actions before resume
- `/schedules` lists owned scheduled work and exports recent run and delivery records
- mentions
- replies
- `/nth`
- `/debug toggle` / `/debug logs`
- `/index`
- `/access`
- `/settings`

Interaction entrypoint:

- `src/discord/events/interaction/interactionCreate.event.ts`

Command registry/loader:

- `src/discord/loaders/commandLoader.ts`

## Tool Schema Rules (read before touching any tool file)

All tool parameter schemas in `src/tools/` are sent verbatim to every model provider, including Google Gemini. Gemini validates JSON Schema strictly and returns a 400 if any rule is violated. The following constraints are **mandatory** for every schema — violations have caused repeated production outages:

1. **Every `type: "object"` must have a `required` field** — even if no fields are required, include `required: []`. This applies to nested objects at any depth (e.g. sub-objects inside a `cursor` property).
2. **Every `type: "array"` must have an `items` field.**
3. **Opaque passthrough objects must still have `properties: {}` and `required: []`** — never leave a `type: "object"` with nothing but a `description`.
4. Do not use `additionalProperties` — Gemini rejects it.
5. Do not use `null` as a type value.

When adding or editing a tool schema, verify every object at every level of nesting satisfies rules 1–3 before running `npm run check`.

Historical violations found and fixed:
- `getGuildContext` — missing `required: []`
- `listMembers` — missing `required: []`
- `listGuildStructure` — missing `required: []`
- `retrieveMessages` — nested `cursor`, `cursor.history`, `cursor.semantic` objects all missing `required: []`; `cursor.history` also missing `properties: {}`
- `createThread` — numeric `enum: [60, 1440, 4320, 10080]` on `auto_archive_duration` (type: "number"); Gemini rejects numeric enums — use string `enum` or drop `enum` entirely and document valid values in the description

## Safe Change Sequence

1. Update the smallest responsible runtime module.
2. Update prompts if the runtime contract changed.
3. Update `AGENTS.md` and the relevant docs.
4. Update or add tests.
5. Run `npm run check`.

Use `npm run test:live` for opt-in real-model verification when prompt or orchestration changes need behavior validation beyond deterministic mocks.

Do not reintroduce giant inline prompts, fake web-search claims, refusal-first grounding, or uncontrolled tool loops.

Capability approval rules live in `access.rules`. Pass the actual tool ID through `AuthorizeAction` at preview and dispatch. Deny overrides Ask, which overrides Allow; no rule raises a grant ceiling. Permitted reads never prompt. Rules are operator-configured through `/access` and inspected with grants.

Named collaborators are steering-only for public guild tasks. Preserve authenticated contributor IDs through persistence and require owner approval for changes after collaborative steering, including resumed tasks. Private tasks, exports and approvals remain owner-only.

Message updates enter through `messageUpdate.event.ts` and the shared ingestion lock. Preserve revision-bearing Discord source URLs through task provenance, derived content and dreaming; do not reconstruct a bare message URL when `sourceUrl` is available. Old edit timestamps cannot replace newer indexed evidence. Deleted messages invalidate every revision.

## Conversational task discovery

`task_search` and `task_control` let the model resolve owned work from ordinary mentions and replies. There is no deterministic intent router. IDs are internal tool arguments; users can describe their previous request. Searches are paginated, actor/location scoped and exclude private or unreadable work before returning descriptions. Control calls recheck ownership and admission. Stops are cooperative, steering forwards the actual user instruction, and continuation passes through Runtime's existing ownership, source, unresolved-action and approval checks. Completed work can reopen explicitly for revision. External actions retain their own approval policy.

## Command reply visibility

`/settings`, `/costs`, `/tasks` summaries, `/steer` and `/stop` now default to channel-visible replies and support `ephemeral:true`. Public task summaries filter private tasks and recheck source audiences. Skills, memories, schedules, access and task exports also default to channel-visible replies with an optional `ephemeral:true`. The settings cost view updates the existing message. Visibility does not change admission, operator checks or component ownership.

## Default voice

Sophia's shared voice lives in `resources/prompts/system/personality.md` and is embedded into the unified runtime prompt. Keep the warmth, composure, occasional dry wit and concise conversation of the earlier default. Register settings do not create separate identities. No em/en dashes in generated prose, canned preambles, habitual closing offers, or forced follow-up questions. Preserve exact source quotations and code. Detailed tasks are not subject to a rigid conversational length cap. Do not restore the obsolete competing personality overrides.


Runtime chronology: the prompt includes the current UTC date and time. Channel context, prior conversation turns, reused evidence and fallback synthesis retain source timestamps. Loaded history is not necessarily current. Latest-message claims require chronological retrieval with newer-range coverage; incomplete results must be described as the latest found in the inspected scope. Offline presence does not establish invisible status.

Cancellation propagates to active chat-completions/Responses HTTP requests and single/batch approval waits. Approval signals are runtime-owned, separate from serialized requests. Cancelling removes pending controls and settles the approval record without dispatch; an already submitted external action still needs its receipt or verification.

The Discord client subscribes to GuildMessages and DirectMessages. Thread replies require SendMessagesInThreads, while ordinary channel replies require SendMessages. Authorized accounts can inspect skills and memories without legacy command visibility; adoption remains operator-only.

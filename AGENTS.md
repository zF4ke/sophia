# Sophia Agent Handbook

Sophia is a Discord assistant with one conversation runtime, one retrieval pipeline, and one local runtime state store.

This file is the engineering handoff for the current runtime.

## Canonical Docs

- `docs/architecture.md`
- `docs/agent-loop.md`
- `docs/memory-indexing.md`
- `docs/prompt-catalog.md`
- `docs/commands-and-admin.md`
- `docs/cleanup-migration.md`
- `docs/testing.md`
- `docs/ui.md`
- `docs/how-sophia-works.md`
- `docs/feature-user-stories.md`

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
- tool-call and latency budgets
- context overflow pruning (Tier-1 truncation + Tier-2 compaction)
- doom-loop detection (identical tool calls → nudge → force finish)
- progress-required tracking (long tasks only)
- refusal prevention for ordinary conversation

Intent policy:
- The active runtime is model-led inside `runtime/agent_loop`.
- Do not reintroduce a separate deterministic intent router or argument-enrichment layer unless there is a clear, measured need.

## Stable Tool Contract

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
- `start_long_task`
- `note_add`
- `note_list`
- `note_clear`
- `plan_update`
- `web_search`
- `fetch_url`
- `memory_search`
- `memory_remember`
- `workflow_create`
- `workflow_list`
- `workflow_run`

Tool schemas are defined in `src/runtime/toolSchemas.ts`.

The stable code catalog is:

- `src/shared/discordTools.ts`

If a capability id changes, update:

- `src/shared/discordTools.ts`
- `src/runtime/toolSchemas.ts`
- `src/capabilities/CapabilityRegistry.ts`
- `resources/prompts/runtime/agent_loop.md`
- the affected tests

Mutating capability output contract (required for all current and future `write`/`destructive` tools):
- Return identifier-rich `data` fields that are directly reusable in user replies (for Discord resources include both raw IDs and mention-ready fields, e.g. `channelId` and `channelMention: <#id>`).
- Include the same concrete identifier in `summary` (not only display names), so the model can produce accurate post-action confirmations.
- Prefer stable identifiers over names when they differ in reliability.

## Prompt Ownership

Runtime prompts live under `resources/prompts/` and stay external to code:

- `resources/prompts/system/base.md`
- `resources/prompts/system/personality.md`
- `resources/prompts/system/personality_mixed_override.md` (applied when `personality = "mixed"`)
- `resources/prompts/system/personality_classic_override.md` (applied when `personality = "classic"`)
- `resources/prompts/runtime/agent_loop.md`

Personality mode is selectable via `/settings` → Personalidade. Three modes: `default` (baseline), `mixed` (recommended — sharper, evidence-first, low filler), `classic` (legacy dominant persona).

The stable prompt catalog is:

- `src/shared/promptCatalog.ts`

If a prompt contract changes, update:

- the prompt file
- `src/runtime/contracts.ts`
- `src/runtime/Runtime.ts`
- `docs/prompt-catalog.md`
- related tests

## Models And Config

Editable runtime config lives in:

- `resources/models/model-profiles.json`
- `storage/settings.json` (managed by `SettingsService`)
- `.env` (secrets only: `DISCORD_TOKEN`, `OPENROUTER_API_KEY`)
- `src/app/AppConfig.ts`

All runtime tuning knobs (tool calls, latency budget, retrieval limits, etc.) live in `SettingsService` defaults and `storage/settings.json`. Do not use environment variables for runtime config.

OpenRouter remains the only model-provider surface. Do not hardcode models in source.

## Memory And Storage

Local libSQL storage is the source of truth for Discord retrieval state and runtime traces.

- New messages are ingested from `src/discord/events/message/messageCreate.event.ts`
- Backfill and repair flows are exposed through `src/discord/commands/system/index.command.ts`
- Operational storage is local and disposable
- Bot settings are persisted in `storage/settings.json`

Do not reintroduce reusable grounded-context caches or tool-result caches into the active runtime path.

## Commands

Core supported commands and entrypoints:

- `/talk`
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

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
- context overflow pruning
- refusal prevention for ordinary conversation

Intent policy:
- The active runtime is model-led inside `runtime/agent_loop`.
- Do not reintroduce a separate deterministic intent router or argument-enrichment layer unless there is a clear, measured need.

## Stable Tool Contract

These capability ids are prompt- and runtime-stable:

- `retrieve_messages`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`

Tool schemas are defined in `src/runtime/toolSchemas.ts`.

The stable code catalog is:

- `src/shared/discordTools.ts`

If a capability id changes, update:

- `src/shared/discordTools.ts`
- `src/runtime/toolSchemas.ts`
- `src/capabilities/CapabilityRegistry.ts`
- `resources/prompts/runtime/agent_loop.md`
- the affected tests

## Prompt Ownership

Runtime prompts live under `resources/prompts/` and stay external to code:

- `resources/prompts/system/base.md`
- `resources/prompts/system/personality.md`
- `resources/prompts/runtime/agent_loop.md`

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
- `/find`
- `/nth`
- `/debug toggle` / `/debug logs`
- `/index`
- `/access`
- `/settings`

Interaction entrypoint:

- `src/discord/events/interaction/interactionCreate.event.ts`

Command registry/loader:

- `src/discord/loaders/commandLoader.ts`

## Safe Change Sequence

1. Update the smallest responsible runtime module.
2. Update prompts if the runtime contract changed.
3. Update `AGENTS.md` and the relevant docs.
4. Update or add tests.
5. Run `npm run check`.

Use `npm run test:live` for opt-in real-model verification when prompt or orchestration changes need behavior validation beyond deterministic mocks.

Do not reintroduce giant inline prompts, fake web-search claims, refusal-first grounding, or uncontrolled tool loops.

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

The active flow is:
1. Normalize the incoming turn.
2. Resolve the canonical conversation key.
3. Load checkpoint state and recent local runtime state.
4. Let the model plan direct conversation vs current-guild retrieval.
5. Retrieve cached Discord evidence first.
6. Escalate to live Discord history when cache evidence is weak.
7. Judge whether the evidence is enough to continue, answer, or ask a targeted follow-up.
8. Synthesize the final response.
9. Persist runtime and trace data locally.

The runtime is model-led by default and keeps only narrow guardrails:
- exact-id structural shortcuts
- capability validation
- repeated-call protection
- budget limits
- refusal prevention for ordinary conversation

## Stable Tool Contract

These capability ids are prompt- and runtime-stable:

- `retrieve_messages`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`

The stable code catalog is:

- `src/shared/discordTools.ts`

If a capability id changes, update:

- `src/shared/discordTools.ts`
- `src/capabilities/CapabilityRegistry.ts`
- `resources/prompts/runtime/plan_turn.md`
- `resources/prompts/runtime/select_next_step.md`
- `resources/prompts/runtime/judge_evidence.md`
- `resources/prompts/runtime/synthesize_answer.md`
- the affected tests

## Prompt Ownership

Runtime prompts live under `resources/prompts/` and stay external to code:

- `resources/prompts/system/base.md`
- `resources/prompts/runtime/plan_turn.md`
- `resources/prompts/runtime/select_next_step.md`
- `resources/prompts/runtime/judge_evidence.md`
- `resources/prompts/runtime/synthesize_answer.md`
- `resources/prompts/runtime/debug_summary.md`
- `resources/prompts/guards/insufficient_evidence.md`

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
- `.env`
- `src/app/AppConfig.ts`

OpenRouter remains the only model-provider surface. Do not hardcode models in source.

## Memory And Storage

Local libSQL storage is the source of truth for Discord retrieval state and runtime traces.

- New messages are ingested from `src/discord/events/message/messageCreate.event.ts`
- Backfill and repair flows are exposed through `src/discord/commands/system/index.command.ts`
- Operational storage is local and disposable
- LangGraph checkpoints are kept in a separate local SQLite file

Do not reintroduce reusable grounded-context caches or tool-result caches into the active runtime path.

## Commands

Core supported commands and entrypoints:

- `/talk`
- mentions
- replies
- `/find`
- `/nth`
- `/debug`
- `/index`
- `/access`
- `/cache`

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

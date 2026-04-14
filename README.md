# Sophia

Sophia is a Discord assistant built around one conversational runtime and one unified Discord retrieval pipeline.

## Public Surface

Main conversation entrypoints:
- `/talk`
- mentions
- replies

Specialized workflow:
- `/find`

Operator surfaces:
- `/nth`
- `/index`
- `/debug`
- `/access`
- `/settings`

## What Sophia Does

Sophia is conversational first. When a turn does not need Discord evidence, she keeps the conversation moving instead of shutting it down.

When a turn does depend on Discord, she uses a cache-first retrieval pipeline:
1. search local indexed Discord messages
2. if evidence is weak, refresh from live Discord history behind the scenes
3. ingest the refreshed messages locally
4. retry retrieval on the enriched cache

That local storage is a Discord retrieval cache plus runtime state. It is not a separate memory-search product.

Runtime budget defaults such as tool-call limits and latency budget are configurable through `/settings` or `.env` overrides.

## Runtime Architecture

Sophia uses a while-loop with native function calling. One unified system prompt (`runtime/agent_loop`) gives the model the question, context, and available tools. The model calls tools iteratively and calls `finish` when it has an answer.

There is no separate planner, step selector, or evidence judge. The model handles all decisions inside the tool-calling loop. The runtime enforces:
- capability validation (tool name must be registered)
- repeated-call protection (same arguments blocked)
- tool-call budget and latency budget
- context overflow pruning
- refusal prevention for ordinary conversation

If the model never calls `finish`, the runtime produces a conversational fallback.

## Conversation Continuity

Conversation identity is reply-chain first:
- a native Discord thread keeps its own conversation key
- replies to Sophia reuse the stored conversation thread when one exists
- replies without a stored Sophia anchor fall back to the first real anchor message
- otherwise Sophia uses a shared channel-level conversation key

This lets `/talk`, mentions, and replies behave like one continuous conversation even when multiple users join the same chain.

## Grounding And Recovery

Sophia should not dead-end a turn with a bare refusal when grounding is weak. The runtime is designed to:
- give the best-effort interpretation from current context
- mark uncertainty clearly when needed
- ask a targeted follow-up or continue the retrieval path

That applies to casual conversation, follow-up questions, and partially grounded channel questions.

## Stable Capabilities

The current stable capability ids are:
- `retrieve_messages`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`

These are exposed to the model as native function-calling tools via `src/runtime/toolSchemas.ts`.

`retrieve_messages` is the main Discord evidence path. `resolve_member_identity`, `list_guild_structure`, and `resolve_channel_targets` are the current-guild discovery layer. The remaining capabilities are live metadata lookups.

## Storage

Active runtime storage lives under `storage/runtime/` and is disposable local state.

Current storage roles:
- operational runtime state and traces
- local message cache and indexing state
- conversation state (SQLite)

Bot settings are stored in `storage/settings.json` and managed through `/settings` or `SettingsService`.

## Documentation

Key docs:
- `AGENTS.md`
- `docs/architecture.md`
- `docs/agent-loop.md`
- `docs/memory-indexing.md`
- `docs/prompt-catalog.md`
- `docs/commands-and-admin.md`
- `docs/cleanup-migration.md`
- `docs/how-sophia-works.md`
- `docs/feature-user-stories.md`
- `docs/testing.md`
- `docs/ui.md`

## Verification

Run:
```bash
npm run check
```

Optional live-model suite:
```bash
LIVE_MODEL_TESTS=1 OPENROUTER_API_KEY=... npm run test:live
```

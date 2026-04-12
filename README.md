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

`/ask` and `/context` are retired. Sophia now treats `/talk`, mentions, and replies as one conversation surface.

## What Sophia Does

Sophia is conversational first. When a turn does not need Discord evidence, she should keep the conversation moving instead of shutting it down.

When a turn does depend on Discord, she uses a cache-first retrieval pipeline:
1. search local indexed Discord messages
2. if evidence is weak, refresh from live Discord history behind the scenes
3. ingest the refreshed messages locally
4. retry retrieval on the enriched cache

That local storage is a Discord retrieval cache plus runtime state. It is not a separate memory-search product.

Runtime budget defaults such as tool-call limits, research-pass count, and latency budget are documented in [.env.example](.env.example).

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

`retrieve_messages` is the main Discord evidence path. `resolve_member_identity`, `list_guild_structure`, and `resolve_channel_targets` are the current-guild discovery layer. The remaining capabilities are live metadata lookups.

## Planning Model

Sophia is model-led by default.

The planner and next-step selector get:
- the question
- trigger type
- reply context
- recent turns
- recent channel context
- active resolved member/channel targets
- the capability registry

The runtime still enforces:
- capability validation
- exact-id structural shortcuts
- repeated-call protection
- tool and latency budgets
- refusal prevention for ordinary conversation

Goal shifting is model-led. If `plan_turn` marks a turn as `continuation=false`, Sophia resets active scoped targets and carried evidence before the research loop. This prevents stale scope leakage from prior tasks when the user changes objective mid-thread.

If model planning fails, Sophia falls back to a small generic current-guild recovery ladder instead of brittle language-specific routing.

## Runtime Tuning

You can tune context and evidence carry-over behavior from env values:
- `RUNTIME_CONTEXT_MAX_CARRIED_EVIDENCE_ITEMS`: max reconstructed evidence items carried from recent tool runs.
- `DEBUG_CONTEXT_PREVIEW_MAX_EVIDENCE_ITEMS`: max evidence items rendered in debug context preview.
- `TOOL_RESOLVE_CHANNEL_TARGETS_MAX_EVIDENCE_ITEMS`: max channel/category metadata evidence items emitted by `resolve_channel_targets`.
- `TOOL_RETRIEVE_MESSAGES_MAX_HISTORY_EVIDENCE_ITEMS`: max history rows converted into evidence per `retrieve_messages` run.
- `TOOL_RETRIEVE_MESSAGES_MAX_SEMANTIC_EVIDENCE_ITEMS`: max semantic rows converted into evidence per `retrieve_messages` run.
- `TOOL_RETRIEVE_MESSAGES_MAX_EVIDENCE_CONTENT_CHARS`: max characters kept per retrieved evidence snippet.

For category and channel questions, the runtime now prefers:
1. resolve the likely target
2. inspect the matched structure
3. retrieve scoped messages from the resolved child channels

## Storage

Active runtime storage lives under `storage/runtime/` and is disposable local state.

Current storage roles:
- operational runtime state and traces
- local message cache and indexing state
- LangGraph checkpoints

Legacy cache and storage artifacts are not part of the active runtime path.

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

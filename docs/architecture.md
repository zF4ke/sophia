# Architecture

Sophia runs on one conversational runtime and one unified Discord retrieval pipeline.

## Main Blocks

- `runtime`
  Graph orchestration, planning, bounded loop control, answer synthesis, debug state, and persistence.
- `conversation`
  Canonical conversation identity, reply-chain continuity, and turn normalization.
- `retrieval`
  Cache-first Discord message retrieval with automatic live refresh when the cache is weak.
- `capabilities`
  Registry-driven capability manifests and handlers.
- `memory`
  Local runtime state, message cache, and trace storage.
- `integrations/discord`
  Thin Discord adapters that convert commands, mentions, and replies into generic runtime inputs.
- `observability`
  Debug rendering and trace capture.
- `security`
  Access, moderation, and command policy.

## Core Principles

- One conversation surface: `/talk`, mentions, and replies all route to the same runtime.
- One retrieval surface: cached Discord evidence first, live Discord refresh second.
- Conversation identity is reply-chain first, then native thread, then channel fallback.
- Weak grounding should lead to best-effort continuation or a targeted follow-up, not a dead-end refusal.
- Capability execution is registry-driven, not hardcoded per tool in the core runtime.
- `retrieve_messages` is the main message-evidence capability. `resolve_member_identity` handles exact member and bot resolution with same-guild historical fallback. `list_guild_structure` and `resolve_channel_targets` provide current-guild discovery. `get_member_profile` returns rich profile data including roles, join date, account creation date, nickname, bot status, Nitro/premium status, and avatar. `list_members` supports offset-based pagination (default page size 20) with an optional name/username fragment filter — omitting the filter returns all members. `get_guild_context` provides live guild-level metadata.
- `/find` stays separate as a specialized retrieval workflow built on the same primitives.

## Active Runtime Flow

1. Turn normalization
2. Conversation identity resolution
3. Checkpoint loading
4. Local message-cache loading
5. Turn planning
6. Mode routing
7. Bounded retrieval loop when needed
8. Evidence judgment
9. Conversational answer synthesis
10. Runtime and trace persistence

## Conversation Identity

Conversation keys are built to avoid nested reply-chain drift.

The active order is:
- native Discord thread
- stored Sophia reply-chain anchor
- first real anchor message for a new reply chain
- shared channel fallback

Trigger type is metadata only. It does not define the conversation key.

## Retrieval Model

Sophia does not treat local storage as a separate memory-search product.

The retrieval path is:
1. search local indexed Discord messages
2. if evidence is weak, refresh likely live Discord history behind the scenes
3. ingest the refreshed messages locally
4. retry retrieval on the enriched cache

That keeps Discord search cheap, durable, and continuously improving.

## Storage

Active runtime storage lives under `storage/runtime/` and is disposable local state.

It stores:
- runtime runs
- tool runs
- trace events
- message cache and indexing state
- conversation state
- LangGraph checkpoints

If runtime storage becomes incompatible, the code now prefers reset-and-rebuild over leaking SQL errors into live replies.

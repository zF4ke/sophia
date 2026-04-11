# Agent Loop

Sophia uses one bounded graph runtime with one retrieval loop.

## Loop Shape

1. `ingest_turn`
2. `load_context`
3. `load_checkpoint`
4. `load_memory`
5. `plan_turn`
6. `route_mode`
7. `run_research_loop` when needed
8. `synthesize_answer`
9. `persist_run`

## Conversation Entry

All conversational entrypoints feed the same runtime:
- `/talk`
- mentions
- replies

Replies add referenced-message context, but they do not fork into a separate runtime.

## Capability Model

Planner-visible capabilities:
- `retrieve_messages`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`

`retrieve_messages` is the main Discord evidence path. It is now history-first and lane-based:
- ordered scoped history messages are the default lane
- semantic matches are a second lane from the same scoped channels
- retrieval can continue across turns through a persisted scoped session without rereading duplicate messages

`resolve_member_identity`, `list_guild_structure`, and `resolve_channel_targets` are the current-guild discovery layer. The others are live metadata capabilities.

The runtime chooses capabilities from the registry. The planner is model-led by default, and the core loop should not grow language-specific routing or tool-specific branching for each new capability.

## Stop Policy

Hard limits:
- max tool calls
- max research passes
- repeated-call guard
- latency budget

Soft exit:
- evidence sufficient
- no useful next step
- confidence plateau
- best-effort continuation preferred over a dead-end refusal

## Message Evidence Rule

Questions like "what did X say" or "what happened in channel Y" require message evidence from `retrieve_messages`. Live member or guild metadata alone is not enough.

For channel and category questions, the expected evidence order is:
1. resolve the target scope
2. inspect structure if category expansion is needed
3. read scoped history
4. supplement with scoped semantic matches only when needed

When evidence is weak, the runtime should continue the conversation with the best grounded interpretation it can produce, then ask a targeted follow-up or continue retrieval instead of stopping cold.

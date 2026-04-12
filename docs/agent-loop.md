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

`load_memory` now restores not only prior turn summaries and active retrieval session metadata, but also reusable evidence reconstructed from recent persisted tool outputs.

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

Continuation inputs are intentionally gated:
- cursor + excluded message ids are reused only for explicit continuation intent
- fresh follow-up turns in the same scope do not automatically inherit dedupe exclusions

For strict scoped reads (author/time bounded), retrieval performs guarded empty-result recovery:
- retry once without excluded ids
- retry once without cursor when needed
- record diagnostics in tool output and debug timeline

`resolve_member_identity`, `list_guild_structure`, and `resolve_channel_targets` are the current-guild discovery layer.

`get_member_profile` returns rich profile data surfaced as evidence content: display name, username, nickname, roles, join date, account creation date, bot status, Nitro/premium status, pending status, and avatar URL. This makes profile comparisons (e.g. "who joined first?") answerable from evidence alone.

`list_members` supports offset-based pagination with a default page size of 20. An optional `filters` parameter narrows results by name/username fragment; omitting it returns all guild members in pages. When more members exist beyond the current page, a `hasMore` flag and pagination hint are included in evidence so the model can request the next page.

The runtime chooses capabilities from the registry. The planner is model-led by default, and the core loop should not grow language-specific routing or tool-specific branching for each new capability.

`candidateCapabilities` from `plan_turn` is initial planning guidance surfaced to the step planner in its prompt. It is not a hard gate. `select_next_step` may choose any capability from the registry based on what it has learned so far, regardless of what was initially planned. The same capability may be called multiple times with different arguments when the model determines that is needed (e.g. `get_member_profile` once per ambiguous member).

The runtime provides argument enrichment (resolved IDs, cursors, time bounds) to help the model execute correctly. It does not override the model's tool choice with pre-flight redirects.

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

Safety override:
- if confidence is `insufficient` and there is no strong message evidence, synthesis must avoid speculative factual/entity claims.

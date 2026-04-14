# Agent Loop

Sophia uses a while-loop with native function calling. One unified system prompt gives the model context and tools. The model calls tools iteratively and calls `finish` when it has an answer.

## Loop Shape

1. Ingest turn (normalize input, resolve conversation key)
2. Load memory (recent turns, channel context, prior evidence from persisted tool runs)
3. Build system prompt (`runtime/agent_loop` with all context injected)
4. Enter while-loop: model receives messages and available tools via `generateWithTools`
5. Model calls Discord tools → runtime executes them → results fed back as tool messages
6. Model calls `finish` → loop exits with the answer
7. Persist runtime run, tool runs, and trace events

If the model never calls `finish`, the runtime produces a conversational fallback based on whatever evidence was collected.

## Conversation Entry

All conversational entrypoints feed the same runtime:
- `/talk`
- mentions
- replies

Replies add referenced-message context, but they do not fork into a separate runtime.

## Capability Model

The model sees these capabilities as native function-calling tools:
- `retrieve_messages`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`
- `finish` (delivers the final answer and exits the loop)

Tool schemas are defined in `src/runtime/toolSchemas.ts`. Capability handlers are registered in `src/capabilities/CapabilityRegistry.ts`.

`retrieve_messages` is the main Discord evidence path. It is history-first and lane-based:
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

`get_member_profile` returns rich profile data surfaced as evidence content: display name, username, nickname, roles, join date, account creation date, bot status, Nitro/premium status, pending status, and avatar URL.

`list_members` supports offset-based pagination with a default page size of 20. An optional `filters` parameter narrows results by name/username fragment; omitting it returns all guild members in pages.

The model chooses which tools to call and in what order. It may call the same tool multiple times with different arguments (e.g. `get_member_profile` once per ambiguous member). The runtime validates tool names, executes capabilities, and enforces budgets, but it does not enrich or rewrite the model's arguments.

## Runtime Guardrails

Hard limits:
- max tool calls per turn
- repeated-call guard (same tool + same arguments blocked)
- latency budget
- context overflow pruning (old tool outputs are pruned when approaching the context window)

## Context Retention

Sophia can read through thousands of stored Discord messages across multiple `retrieve_messages` calls, but it does not keep all of those rows in the live model context at once.

What is kept in the prompt:
- the unified system prompt
- the current user message
- recent conversation turns (`maxPriorTurns`)
- recent ambient channel messages (`maxChannelMessages`)
- prior evidence reconstructed from persisted tool runs (`maxEvidenceSlice`)
- the latest tool interaction inside the active loop

What is trimmed or pruned:
- old tool outputs are pruned first when prompt usage reaches about 80% of the selected model profile context window
- each individual tool result injected back into the loop is capped before insertion
- older persisted evidence outside the configured carry-over slice is dropped before the next turn starts

The practical effect is: retrieval can search very deep history, but the prompt keeps only a bounded working set plus the newest loop messages.

Soft exit:
- model calls `finish` when it has enough
- fallback answer when model produces no tool calls and no text

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

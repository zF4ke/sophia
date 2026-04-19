# Agent Loop

Sophia uses a while-loop with native function calling. One unified system prompt gives the model context and tools. The model calls tools iteratively and calls `finish` when it has an answer.

## Loop Shape

1. Ingest turn (normalize input, resolve conversation key)
2. Load memory (recent turns, channel context, prior evidence from persisted tool runs)
3. Build system prompt (`runtime/agent_loop` with all context injected)
4. Enter while-loop: model receives messages and available tools via `generateWithTools`
5. Model calls tools → runtime validates and executes them → results fed back as tool messages
6. Write/destructive tool calls pass through the approval gate before execution
7. Model calls `finish` → loop exits with the answer
8. Persist runtime run, tool runs, and trace events

If the model never calls `finish`, the runtime produces a conversational fallback based on whatever evidence was collected.

## Conversation Entry

All conversational entrypoints feed the same runtime:
- `/talk`
- mentions
- replies

Replies add referenced-message context, but they do not fork into a separate runtime.

## Capability Model

The model sees these capabilities as native function-calling tools:

### Read & Discovery Tools
- `retrieve_messages` — cache-first scoped history search with semantic lane, pagination, time bounds, author filters
- `resolve_member_identity` — resolve members by ID, username, nickname, display name (batch)
- `list_guild_structure` — list readable channels/categories + cached-only remembered entries
- `resolve_channel_targets` — resolve channel/category names, IDs, mentions; expands categories to child channels
- `get_member_profile` — rich profile data (roles, join date, account created, Nitro, avatar, etc.)
- `list_members` — live guild members with offset pagination and optional name filter
- `get_guild_context` — guild-level metadata (name, member count, etc.)
- `get_role_info` — role details (members, permissions, color, position)
- `list_threads` — active + recently archived threads in a channel
- `read_thread_messages` — read messages from a specific thread

### Utility Tools
- `measure_text_length` — count characters, words, lines
- `evaluate_math` — safe arithmetic evaluation

### Write Tools (require admin approval)
- `create_channel` — create a new channel
- `create_category` — create a new category
- `create_thread` — create a thread in a channel
- `move_channel` — move a channel to a different category/position
- `manage_member_roles` — add/remove roles from a member
- `send_message` — send a message to a channel or thread

### Destructive Tools (require approval + confirmation)
- `clear_messages` — delete messages from a channel
- `delete_channel` — permanently delete a channel

### Control
- `finish` — delivers the final answer and exits the loop
- `start_long_task` — declares the current task needs an elevated tool-call/latency budget (runtime-intercepted, never dispatched to capability layer)

Tool schemas are defined in `src/runtime/toolSchemas.ts`. Capability handlers are registered in `src/capabilities/CapabilityRegistry.ts`.

## Approval Gate

Write and destructive tool calls do not execute immediately. They pass through an approval gate:

- **Write tools**: An approval card is shown to the admin with Aceitar (approve) / Recusar (deny) buttons. If `autoApproveWrites` is enabled in settings, these are auto-approved.
- **Destructive tools**: Always require explicit admin approval via an approval card, followed by a confirmation dialog ("Esta ação é destrutiva. Tens a certeza?").
- **Batch destructive**: Multiple destructive calls in the same model response are grouped into a single batch approval card. The batch card shows all pending actions and allows approve-all, deny-all, or approve-by-category (grouped by Discord parent category).

Approval cards include:
- Tool name and description
- Side-effect level badge (yellow for write, red for destructive)
- Timeout countdown (configurable via `approvalTimeoutMs`)
- Post-action status icons (✅ approved, ❌ denied, ✏️ corrected, ⏳ timed out, 🛑 stopped)

The "Recusar e corrigir" button opens a modal where the admin can explain what should be done differently. That feedback is returned to the model.

## Retrieval Deep Dive

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

## Runtime Guardrails

Hard limits:
- max tool calls per turn (configurable, default 6, range 2–30; raiseable to 200 via `start_long_task`)
- repeated-call guard (same tool + same arguments blocked)
- latency budget (configurable, default 20s, range 10s–5m; raiseable to 600s via `start_long_task`)
- context overflow pruning (old tool outputs are pruned when approaching the context window)

### Stall Guard

When the model calls `finish`, the runtime checks for empty-promise stalling: if the answer contains a promise phrase (PT/EN) but no productive tool ran and no productive evidence was produced in the turn, the finish is rejected once and the model is told to call tools instead. At most 1 correction per turn.

### Long-Task Budget

The `start_long_task` tool allows the model to self-declare that a task needs more budget. The runtime intercepts the call (it is never dispatched to the capability layer) and raises `maxToolCalls` and `maxLatencyBudgetMs` up to hard caps (200 calls, 600s). Idempotent per turn.

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

## Mutating Tool Output Contract

For all `write` and `destructive` capabilities (current and future), tool output must be mention-ready so post-action confirmations can reference the exact affected resource.

Required shape and behavior:
- `summary` must include a concrete resource identifier, not only a display name.
- `data` must include stable IDs and a mention-ready field when relevant (example for channels: `channelId` and `channelMention` with `<#id>`).
- Final user-facing confirmations should reuse these identifiers from tool output to avoid ambiguity.

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

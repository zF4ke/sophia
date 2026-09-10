# Architecture

Sophia runs on one conversational runtime and one unified Discord retrieval pipeline.

## Main Blocks

- `runtime`
  While-loop agent with native function calling, context management, answer synthesis, debug state, and persistence.
- `conversation`
  Canonical conversation identity, reply-chain continuity, and turn normalization. One shared channel conversation by default; parallel questions in the same channel fork from the last committed state instead of sharing mutable in-flight state.
- `retrieval`
  Cache-first Discord message retrieval with automatic live refresh when the cache is weak.
- `capabilities`
  Registry-driven capability manifests and handlers. 49 tools across read, write, destructive, web, memory, and workflow tiers (GLM 5.3 Flash default, 1.3M context).
- `tool execution`
  One executor validates model arguments, invokes capabilities, normalizes failures, extracts evidence, and returns the durable tool-run shape.
- `approval`
  Gate layer for write and destructive tool calls. Single-item cards for individual actions; batch cards for grouped destructive actions organized by Discord category.
- `memory`
  Local runtime state, message cache, and trace storage.
- `integrations/discord`
  Thin Discord adapters that convert commands, mentions, and replies into generic runtime inputs.
- `observability`
  Debug rendering and trace capture.
- `security`
  Access, moderation, rate limiting, and command policy.

## Core Principles

- One conversation surface: `/talk`, mentions, and replies all route to the same runtime.
- One retrieval surface: cached Discord evidence first, live Discord refresh second.
- Conversation identity is reply-chain first, then native thread, then channel fallback.
- Weak grounding should lead to best-effort continuation or a targeted follow-up, not a dead-end refusal.
- The model decides what to do inside a while-loop with native function calling. The runtime enforces budgets and guardrails but does not pre-plan or redirect tool choices.
- OpenRouter chat requests sort eligible providers by highest throughput while retaining provider fallbacks.
- Model calls use a fixed 10-minute HTTP timeout and disable SDK retries. The gateway retries provider errors (429/5xx) with short backoff, and a non-default model that keeps failing falls back to the default profile's model for that call. OpenRouter still handles provider fallback.
- Turns have no wall-clock cap: the loop is bounded by the tool-call budget, per-request timeouts, and the guardrails. Long tasks raise the tool-call cap, not the clock.
- Model profiles with a `baseUrl` (e.g. a local LM Studio server) bypass OpenRouter for chat: no `provider` routing block, no default-model fallback. Embeddings always route through OpenRouter.
- Capability execution is registry-driven, not hardcoded per tool in the core runtime.
- Catalog effects and capability side-effect levels must match. Registry startup fails on duplicate tools or effect drift.
- Write and destructive tool calls pass through an approval gate before execution. Destructive calls additionally require a confirmation dialog. Multiple destructive calls in the same response are batched into one approval card grouped by Discord category.
- Graceful shutdown stops the crawler, gives active Discord replies up to 30 seconds to finish sending, then closes the client.
- `retrieve_messages` is the main message-evidence capability. `resolve_member_identity` handles exact member and bot resolution with same-guild historical fallback. `list_guild_structure` and `resolve_channel_targets` provide current-guild discovery. `get_member_profile` returns rich profile data including roles, join date, account creation date, nickname, bot status, Nitro/premium status, and avatar. `list_members` supports offset-based pagination (default page size 20) with an optional name/username fragment filter — omitting the filter returns all members. `get_guild_context` provides live guild-level metadata. `get_role_info` provides role details. `list_threads` and `read_thread_messages` support thread discovery and reading.

## Active Runtime Flow

1. Turn normalization
2. Conversation identity resolution
3. Load memory (recent turns, channel context, prior evidence from persisted tool runs)
4. Build unified system prompt (`runtime/agent_loop`)
5. While-loop: model calls tools via native function calling, runtime executes them and feeds results back
6. `start_long_task` intercepted if present (raises budget caps; never dispatched to capability layer)
7. Write/destructive tool calls pass through the approval gate before execution
8. On `finish`, stall guard checks for empty-promise answers and re-prompts once if detected
9. Model calls `finish` with the final answer, or runtime produces a conversational fallback
10. Persist runtime run, tool runs, and trace events

## Approval Layer

The approval layer sits between the model's tool call and the capability handler:

- **Write tools** (`create_channel`, `create_category`, `create_thread`, `move_channel`, `manage_member_roles`, `send_message`): require admin approval via an approval card with Aceitar/Recusar buttons. Can be auto-approved when `autoApproveWrites` is enabled.
- **Destructive tools** (`clear_messages`, `delete_messages`, `delete_channel`): always require admin approval plus a confirmation dialog. Multiple destructive calls in the same response are grouped into a batch approval card organized by Discord category.

Approval cards show the tool name, description, side-effect level badge, and timeout countdown. Batch cards include a category select menu when actions span multiple Discord categories.

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

Deep retrieval does not mean deep prompt stuffing. Sophia may inspect thousands of stored messages over multiple tool calls, but only a bounded working set goes into the live model context: recent turns, recent channel context, capped prior evidence, and the latest loop messages. Older tool outputs are pruned when prompt usage approaches the selected model profile context window.

Context management has three tiers (less aggressive since Sophia 4.5):
- **Tier-0 (input compaction)**: bulky prior context (turns, channel context, prior evidence) is summarized once pre-loop if prompt exceeds 55% of the window (90% for 1M models).
- **Tier-1 (pruning)**: oldest tool-result messages are dropped when prompt tokens exceed 85% of the window; preserves last 8 messages (aligns with Tier-2 tail).
- **Tier-2 (compaction)**: when Tier-1 is not enough, the middle block is summarized by `compaction.summarizerModel` (default GLM 5.3 Flash) and replaced with `<compaction_summary>`. Load-bearing scratchpad calls (note_add, plan_update) are preserved verbatim. Requires at least 4 middle messages to avoid trivial summaries.

For long tasks (`start_long_task`), the runtime also provides:
- **Scratchpad tools** (`note_add`, `note_list`, `note_clear`, `plan_update`): per-request notes stored in libSQL. The plan is re-injected every iteration so it survives compaction.
- **Doom-loop detection**: window=6, threshold=4 identical calls → nudge → force finish (less aggressive than before).
- **Progress-required tracking**: threshold=8 non-progress calls (was 5) before nudging — advisory, not blocking.
- **Web tools** (`web_search`, `fetch_url`): internet research with Brave → DuckDuckGo fallback.
- **Long-term memory** (`memory_remember`, `memory_search`): cross-session guild/user memories persisted in `long_term_memories` with an FTS5 keyword index (prefix matching, diacritic-insensitive). A short per-turn memory digest (counts + recent keys) is injected into the system prompt so the model knows the store exists without bloat; `memory_search` returns the full ranked set. User-scoped memories are visible only to their owner.
- **Workflows** (`workflow_create`, `workflow_list`, `workflow_run`, `workflow_delete`): reusable tool chains stored per guild. `workflow_run` loads steps, then the model submits each step through normal tool calls so approval and tracing cannot be bypassed.
- **Artifacts** (`artifact_send`, `artifact_edit`, `src/discord/artifacts/`): interactive Components V2 cards covering the full component surface: text displays, sections with thumbnail or button accessories, media galleries, file cards (re-uploaded attachments), accent color, spoiler, navigation (tab dropdown or pagination), link buttons, custom action rows (buttons and every select type), TTL auto-deletion via a boot-time sweep (`ArtifactStore`), and sandboxed JS handlers (`node:vm`, 100ms, no host globals) that mutate persisted per-card state, reply, send, and re-render on click. Spec validation gates every send, including direct-image-link checks for media. Separate from saved workflows.

## Storage

Active runtime storage lives under `storage/runtime/` and is disposable local state.

It stores:
- runtime runs
- tool runs
- trace events
- message cache and indexing state
- conversation state

Bot settings are stored in `storage/settings.json`.

If runtime storage becomes incompatible, the code prefers reset-and-rebuild over leaking SQL errors into live replies.

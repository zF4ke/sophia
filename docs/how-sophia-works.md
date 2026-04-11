# How Sophia Works

## Overview

Sophia is a conversational-first Discord assistant. She is not a loose prompt harness anymore. She runs as a bounded graph runtime with one conversation system and one Discord retrieval system.

She currently does three main things:
- keep casual conversation moving
- retrieve Discord evidence when a turn depends on server state
- preserve continuity through checkpoints and a local message cache

She does not yet have autonomous long-term memory, a self-updating personality system, or write-side task execution.

## Core Runtime Flow

1. A command, mention, or reply is normalized into one turn format.
2. Sophia resolves a canonical conversation key.
3. She loads checkpoint state for that conversation.
4. She makes cached Discord memory available.
5. She plans whether the turn is direct conversation or bounded retrieval.
6. If retrieval is needed, she runs one capability at a time.
7. She judges whether the evidence is enough.
8. She writes the final answer or asks a targeted follow-up.
9. She persists runtime and trace data.

## How Conversation Continuity Works

Sophia tracks continuity in two layers.

### 1. Checkpoint continuity

This is short-term execution continuity.

Conversation identity is:
- native Discord thread first
- stored Sophia reply-chain anchor second
- first real anchor message for a new reply chain
- shared channel fallback otherwise

That means:
- `/talk` and a later reply to Sophia can stay in the same conversation
- another participant can join the same reply-chain and stay in the same conversation
- a plain new message in the same channel falls back to the shared channel conversation if there is no reply or thread anchor

### 2. Local Discord message cache

This is retrieval continuity.

Sophia stores observed and fetched Discord messages locally so future searches can be answered faster without always hitting Discord first.

## How She Knows Who Is Talking And Who Said What

Sophia keeps requester identity and evidence identity separate.

Requester identity comes from the live Discord trigger:
- for `/talk`, the requester is the interaction user
- for mentions and replies, the requester is the author of the message that triggered Sophia
- the runtime stores the requester's user id and display name in the normalized turn input

Reply identity comes from the referenced Discord message:
- when you reply to a message, Sophia fetches that referenced message
- she stores the referenced message id, author id, author username, author display name, content preview, and jump link
- this lets her know both who is asking and who is being replied to

Evidence identity comes from retrieved Discord data:
- cached or live-fetched message evidence includes author id, author name, channel, timestamp, and jump link
- live member lookups include the resolved member id and display name
- synthesis is supposed to answer from that structured evidence instead of guessing who said what

That distinction matters for user stories like:
- “Who am I?” where the requester must resolve to the speaker, not a random similar member
- “What did Alice say in #reflexoes?” where the answer must come from message evidence authored by Alice
- “What did you think about that?” where Sophia should understand the current requester, the referenced message, and the prior reply-chain context at the same time

## How She Decides What To Do

Sophia decides in stages.

### `plan_turn`

The model first decides whether this is:
- direct conversation
- Discord retrieval

The runtime does not hardcode a large phrase-classification tree anymore. The model sees:
- the question
- the trigger type
- reply context
- recent turns
- recent channel context
- the capability registry
- whether a guild is available

The runtime then applies only narrow guardrails:
- exact-id structural shortcuts
- capability validation
- loop/budget limits
- refusal prevention for ordinary conversation
- a small generic fallback plan if model output is invalid
- short-lived resolved-target carry-over for the active conversation thread

### `run_research_loop`

If she needs Discord evidence, she runs a bounded loop with registry-driven capabilities.

Current planner-visible capabilities:
- `retrieve_messages`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`

For category/channel questions, the intended chain is:
1. resolve the target category/channel
2. inspect the matched guild structure
3. retrieve scoped messages from the resolved child channels

### `judge_evidence`

After each step, Sophia checks whether she has enough evidence to stop.

### `synthesize_answer`

She then turns the result into the final reply or a best-effort conversational follow-up.

## What She Can Do Today

- answer conversational questions directly
- answer server questions with cached or live Discord evidence
- continue reply-chain conversations across participants
- remember prior turns within the same checkpointed conversation
- search cached Discord messages
- automatically fetch more live Discord history when cached evidence is weak
- inspect live member and guild metadata
- resolve exact member, bot, channel, and category ids in the current guild
- distinguish current live guild structure from cached-only remembered entries
- carry short-lived resolved member/channel targets across follow-up turns in the same conversation
- inspect category structure and then retrieve scoped messages from its visible child channels
- run `/find` as a specialized retrieval workflow

## Limitations

Current limitations:
- no autonomous long-term belief store
- no episodic memory layer
- no self-updating personality module
- no write actions like role changes, thread creation, or DM task workflows
- no universal capability composer yet
- retrieval is cache-first Discord search plus targeted live refresh, not a full semantic memory system

## Future Evolution

Planned but not yet implemented:
- continuous long-term memory about people, concepts, and projects
- a separate personality layer that does not contaminate factual memory
- bounded write-side tasks with approvals
- a future composer that can combine specialized workflows and general capabilities under one orchestration layer

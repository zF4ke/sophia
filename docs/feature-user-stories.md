# Feature User Stories

This document is the concrete acceptance layer for Sophia's current runtime. It is intentionally feature-complete for the tools and compositions that exist today.

## Tool Coverage

Current runtime capabilities and workflows:
- `resolve_member_identity`
- `get_member_profile`
- `list_members`
- `resolve_channel_targets`
- `list_guild_structure`
- `retrieve_messages`
- `get_guild_context`
- `/find` specialized workflow

## Single-Tool Stories

### `resolve_member_identity`

- As a user, I can ask who a person, bot, mention, or exact ID is in the current guild.
- Sophia should resolve exact IDs even if the member is not already in the Discord.js cache.
- If the person is no longer a live guild member but exists in same-guild cached history, Sophia may surface that identity as historical rather than current.

### `get_member_profile`

- As a user, I can ask for a member or bot profile and Sophia can return current-guild profile details such as display name and visible roles.
- This tool is for profile metadata, not message-history answers.

### `list_members`

- As an operator or user, I can ask for matching guild members and Sophia can enumerate them from the current guild.
- Exact IDs and filtered lookups should not rely only on cache fragments.

### `resolve_channel_targets`

- As a user, I can mention a channel, category, exact channel ID, or exact category ID and Sophia can resolve the correct current-guild target.
- If I target a category, Sophia should expand it into message-channel ids for later retrieval.

### `list_guild_structure`

- As a user, I can ask what channels/categories exist and Sophia can inspect the readable live guild structure.
- Cached-only remembered channels/categories may also appear, clearly marked as non-live.

### `retrieve_messages`

- As a user, I can ask what someone said or what happened in a channel and Sophia can retrieve message evidence.
- Retrieval should search the local cache first, refresh live Discord history when needed, ingest new messages, and retry.

### `get_guild_context`

- As a user, I can ask about guild-level facts such as server name, member count, or channel count.
- Sophia should answer from current live guild metadata.

### `/find`

- As an operator, I can run `/find` directly with a topic plus optional author/channel/category targeting.
- `/find` should reuse the same shared member/channel/category resolution primitives as the main runtime.

## Composition Stories

### Identity then profile

- User asks: `Quem sou eu?`
- Runtime resolves the requester exactly with `resolve_member_identity`.
- If extra detail is needed, it can follow with `get_member_profile`.
- Sophia answers naturally instead of exposing raw tool output.

### Member + channel + message retrieval

- User asks: `Do que o One Person está falando em #reflexoes?`
- Runtime resolves the speaker with `resolve_member_identity`.
- Runtime resolves the target channel with `resolve_channel_targets`.
- Runtime retrieves scoped message evidence with `retrieve_messages`.
- Sophia answers from those messages, not from profile metadata alone.

### Channel target + guild structure

- User asks: `What channel is 123456789012345678?`
- Runtime resolves the explicit id with `resolve_channel_targets`.
- Runtime loads the guild structure with `list_guild_structure`.
- Sophia answers with the resolved channel and its category/visibility context.

### `/find` with scoped search

- Operator runs `/find topic:\"deployment anxiety\" author:\"One Person\" target:\"#reflexoes\"`
- Workflow resolves author and target scope first.
- Retrieval runs with exact `authorId` and exact resolved `channelIds`.
- Results come back from the same cache-first/live-refresh retrieval stack used by the main runtime.

### Guild overview

- User asks: `How big is this server and what categories does it have?`
- Runtime can combine `get_guild_context` with `list_guild_structure`.
- Sophia answers with current guild metadata plus structure context.

## Behavioral Expectations

- Sophia should stay conversational in the final answer, even when tools were involved.
- Message-history questions must be answered from message evidence, not only member metadata.
- Exact IDs should be first-class inputs for members, bots, channels, and categories.
- Same-guild historical fallbacks must be labeled as historical, not current membership or live structure.
- Tool composition should stay bounded by runtime budgets and repeated-call guards.

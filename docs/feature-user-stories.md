# Feature User Stories

This document is the concrete acceptance layer for Sophia's current runtime. It is intentionally feature-complete for the tools and compositions that exist today.

## Tool Coverage

Current runtime capabilities and workflows:

### Read & Discovery
- `resolve_member_identity`
- `get_member_profile`
- `list_members`
- `resolve_channel_targets`
- `list_guild_structure`
- `retrieve_messages`
- `get_guild_context`
- `get_role_info`
- `list_threads`
- `read_thread_messages`

### Utilities
- `measure_text_length`
- `evaluate_math`

### Write (admin approval required)
- `create_channel`
- `create_category`
- `create_thread`
- `move_channel`
- `manage_member_roles`
- `send_message`

### Destructive (approval + confirmation required)
- `clear_messages`
- `delete_messages`
- `delete_channel`

### Workflows

## Single-Tool Stories

### `resolve_member_identity`

- As a user, I can ask who a person, bot, mention, or exact ID is in the current guild.
- Sophia should resolve exact IDs even if the member is not already in the Discord.js cache.
- If the person is no longer a live guild member but exists in same-guild cached history, Sophia may surface that identity as historical rather than current.

### `get_member_profile`

- As a user, I can ask for a member or bot profile and Sophia can return rich profile evidence: display name, username, nickname, roles, join date, account creation date, bot status, Nitro/premium status, pending status, and avatar URL.
- Profile evidence is surfaced directly in evidence content so the model can compare members (e.g. "who joined first?") without needing raw data access.
- This tool is for profile metadata, not message-history answers.

### `list_members`

- As an operator or user, I can ask for matching guild members and Sophia can enumerate them from the current guild.
- Omitting the `filters` parameter returns all guild members in offset-based pages (default page size 20).
- When more members exist beyond the current page, a `hasMore` flag and pagination hint are included in evidence.
- Exact IDs and filtered lookups should not rely only on cache fragments.

### `resolve_channel_targets`

- As a user, I can mention a channel, category, exact channel ID, or exact category ID and Sophia can resolve the correct current-guild target.
- If I target a category, Sophia should expand it into message-channel ids for later retrieval.

### `list_guild_structure`

- As a user, I can ask what channels/categories exist and Sophia can inspect the readable live guild structure.
- Cached-only remembered channels/categories may also appear, clearly marked as non-live.

### `retrieve_messages`

- As a user, I can ask what someone said or what happened in a channel and Sophia can retrieve message evidence.
- Retrieval should read scoped channel history first, not rely only on a flat semantic search.
- Retrieval should also return scoped semantic matches as a separate lane when they help answer targeted questions.
- Retrieval should search the local cache first, refresh live Discord history when needed, ingest new messages, and retry.
- Retrieval should be resumable across turns with continuation cursors and dedupe protection, so Sophia can keep reading older history without repeating the same messages.
- Retrieval should support normalized before/after time bounds for requests like `until yesterday` or `last week in #atlas`.

### `get_guild_context`

- As a user, I can ask about guild-level facts such as server name, member count, or channel count.
- Sophia should answer from current live guild metadata.

### `get_role_info`

- As a user, I can ask about a specific role's details: members who have it, permissions, color, position, and mentionability.
- Sophia should answer from live role metadata, not cached or guessed data.

### `list_threads`

- As a user, I can ask what threads exist in a channel.
- Sophia should list active and recently archived threads.
- The `include_archived` flag can be toggled to control whether archived threads appear.

### `read_thread_messages`

- As a user, I can ask what was discussed in a specific thread.
- Sophia should read messages from that thread and answer from the evidence.

### `measure_text_length`

- As a user, I can ask Sophia to count characters, words, or lines in a piece of text.
- Sophia should return exact counts without approximation.

### `evaluate_math`

- As a user, I can ask Sophia to calculate arithmetic expressions.
- Sophia should use safe evaluation supporting arithmetic, exponents, sqrt, trig, log, etc.

### `create_channel`

- As an admin, I can ask Sophia to create a new channel.
- Sophia should present an approval card before executing.
- The approval card shows the tool name, description, and a "write" badge.
- If `autoApproveWrites` is enabled, the action executes immediately.

### `create_category`

- As an admin, I can ask Sophia to create a new category.
- Same approval flow as `create_channel`.

### `create_thread`

- As an admin, I can ask Sophia to create a thread in a specific channel.
- Same approval flow as `create_channel`.

### `move_channel`

- As an admin, I can ask Sophia to move a channel to a different category or position.
- Same approval flow as `create_channel`.

### `manage_member_roles`

- As an admin, I can ask Sophia to add or remove roles from a member.
- Same approval flow as `create_channel`.

### `send_message`

- As an admin, I can ask Sophia to send a message to a specific channel or thread.
- Same approval flow as `create_channel`.

### `clear_messages`

- As an admin, I can ask Sophia to delete messages from a channel.
- Sophia must show an approval card with a "destructive" badge.
- After the admin clicks "Aceitar", a confirmation dialog appears: "Esta ação é destrutiva. Tens a certeza?"
- The admin must click "Confirmar" to execute. This cannot be auto-approved.

### `delete_messages`

- As an admin, I can ask Sophia to delete one or more specific messages by ID from a channel (e.g. by replying/quoting a message, or by pointing at content keywords or an author).
- Sophia should first locate the target message IDs with `search_messages` or `retrieve_messages`, never fabricate IDs.
- Same destructive approval + confirmation flow as `clear_messages`.
- Each message is deleted individually, so it works for messages of any age (not subject to the 14-day bulk-delete limit).
- If some IDs fail to delete (already gone, permission denied, etc.), Sophia reports how many succeeded and which failed.

### `delete_channel`

- As an admin, I can ask Sophia to permanently delete a channel.
- Same destructive approval + confirmation flow as `clear_messages`.


## Composition Stories

### Ambiguous identity disambiguation

- User asks: `Qual Drennan é o verdadeiro?`
- Runtime resolves members with `resolve_member_identity` or `list_members`.
- If multiple current-guild members share the same display name, the runtime should fetch a profile for each with `get_member_profile` (once per member, different arguments).
- Sophia compares the profiles (roles, join date, account age, activity history) and makes a specific recommendation instead of asking the user to choose manually.

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

### Category structure + scoped retrieval

- User asks: `Que serviços estão disponíveis nesse servidor?`
- Runtime resolves the likely category or service area with `resolve_channel_targets`.
- Runtime loads the matched guild structure with `list_guild_structure`.
- If the matched target is a category, Sophia expands it to visible child channels.
- Runtime retrieves scoped message evidence from those resolved child channels with `retrieve_messages`.
- If one of those child channels is not indexed locally yet, scoped retrieval should still run and automatically escalate to live Discord fetch for that channel, ingest the messages, and retry.
- Sophia answers from the combination of structure plus scoped messages instead of claiming the category is empty or guessing from a partial alphabetic subset.

### Multi-turn scoped continuation

- User asks: `Me mostra tudo de #atlas até ontem.`
- Runtime resolves `#atlas`, normalizes the time bound, and starts `retrieve_messages` in history-first mode.
- If one pass is not enough, Sophia should keep an active retrieval session with channel scope, anchors, and dedupe state.
- User follows up with `continue` or `de novo`.
- Runtime should continue the same scoped read instead of restarting discovery or rereading duplicate messages.

- Workflow resolves author and target scope first.
- Retrieval runs with exact `authorId` and exact resolved `channelIds`.
- Results come back from the same cache-first/live-refresh retrieval stack used by the main runtime.

### Guild overview

- User asks: `How big is this server and what categories does it have?`
- Runtime can combine `get_guild_context` with `list_guild_structure`.
- Sophia answers with current guild metadata plus structure context.

### Thread discovery and reading

- User asks: `What threads are in #geral and what are they about?`
- Runtime calls `list_threads` to enumerate threads.
- Runtime calls `read_thread_messages` for each relevant thread.
- Sophia summarizes thread topics from message evidence.

### Batch destructive with approval

- Admin says: `Clear messages in #spam and #temp, then delete #old-announcements.`
- Runtime queues `clear_messages` for #spam, `clear_messages` for #temp, and `delete_channel` for #old-announcements.
- All three are grouped into a single batch approval card, organized by Discord category.
- Admin can approve all, deny all, or selectively approve by category.
- If the admin clicks "Recusar e corrigir", a modal opens for correction feedback.

### Write with auto-approve

- `autoApproveWrites` is enabled in settings.
- Admin says: `Create a channel called #project-updates under the Projects category.`
- Runtime calls `create_channel`.
- Because autoApproveWrites is enabled and this is a write (not destructive), the action executes immediately without an approval card.

### Role management after member lookup

- Admin asks: `Give the Moderator role to Drennan.`
- Runtime resolves "Drennan" with `resolve_member_identity`.
- If ambiguous, fetches profiles with `get_member_profile` to disambiguate.
- Runtime calls `manage_member_roles` to add the role.
- Approval card is shown. Admin approves. Role is added.

## Behavioral Expectations

- Sophia should stay conversational in the final answer, even when tools were involved.
- Message-history questions must be answered from message evidence, not only member metadata.
- Exact IDs should be first-class inputs for members, bots, channels, and categories.
- Same-guild historical fallbacks must be labeled as historical, not current membership or live structure.
- Tool composition should stay bounded by runtime budgets and repeated-call guards.
- The model may call any registered capability, in any order, as many times as needed within the tool-call budget.
- If multiple guild members share the same display name, Sophia should fetch a profile for each before answering so she can compare and recommend the right one.
- Category or channel existence alone is not enough to claim what a service does; Sophia should prefer scoped messages when those channels are readable.
- Sophia should not say a category is empty unless the evidence explicitly shows zero visible child channels.
- If a target channel is readable but not indexed, Sophia should still try scoped retrieval and let the retrieval layer do the cache miss -> live fetch -> ingest path automatically.
- For channel-understanding tasks, recent/ordered history is the default evidence lane and semantic matches are supplemental.
- If a large scoped read stops because of budget, Sophia should say so and make it clear that continuation is still possible when that is true.
- Write and destructive tools should only be called when the user explicitly requests an action. Sophia should never speculatively create, delete, or modify server resources.
- Destructive tool calls always require admin approval plus confirmation, regardless of settings.
- Multiple destructive calls in the same response should be batched into one approval card grouped by Discord category.
- After an approved action, Sophia should confirm what was done using the concrete identifiers from tool output (channel mentions, role names, etc.).
- If the model tries to call `finish` with a promise phrase ("vou verificar", "let me check") but no productive tool actually ran, the runtime rejects the finish once and tells the model to call tools instead (stall guard).
- For complex multi-step operations that exceed default budgets, the model should call `start_long_task` early in the turn to raise the tool-call and latency limits (up to hard caps of 200 calls / 600s).

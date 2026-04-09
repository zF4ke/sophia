# Prompt Catalog

Runtime prompts are stored on disk and loaded through `PromptRegistry`.

## Prompt IDs

- `system/base`
- `system/grounded`
- `tasks/classify_request`
- `tasks/plan_discord_search`
- `tasks/synthesize_answer`
- `guards/insufficient_evidence`

The stable code catalog is `src/shared/promptCatalog.ts`.

## Tool-Aware Prompt

`resources/prompts/tasks/plan_discord_search.md` is tool-aware and must stay aligned with:

- `search_messages`
- `read_message_thread`
- `read_channel_summary`
- `list_relevant_channels`
- `get_member_profile`
- `list_members`
- `get_guild_context`

## JSON Contracts

These prompts return strict JSON and must stay synchronized with `src/shared/appTypes.ts`:

- `tasks/classify_request`
- `tasks/plan_discord_search`

If you change the output schema, update the prompt, types, runtime parser, tests, and this file together.

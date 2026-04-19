# Prompt Catalog

Runtime prompts remain external and are loaded through `PromptRegistry`.

## Active Prompt IDs

- `runtime/agent_loop`

The stable code catalog is `src/shared/promptCatalog.ts`.

## Purpose

- `runtime/agent_loop`
  Unified system prompt for the native tool-calling loop. Defines identity, environment, research flow, evidence rules, tool calling rules, and voice.

Legacy prompt files under `resources/prompts/system/` may still exist as reference material, but they are not part of the active runtime prompt catalog.


## Stable Capability Names In Prompt Context

The runtime prompt set expects these capability ids to stay stable:

- `retrieve_messages`
- `search_messages`
- `random_channel_message`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`
- `clear_messages`
- `delete_messages`
- `create_channel`
- `create_category`
- `measure_text_length`
- `evaluate_math`
- `get_role_info`
- `delete_channel`
- `delete_role`
- `create_thread`
- `move_channel`
- `move_category`
- `manage_member_roles`
- `list_roles`
- `list_threads`
- `read_thread_messages`
- `send_message`
- `edit_message`
- `create_role`
- `edit_channel`
- `edit_role`
- `start_long_task`
- `note_add`
- `note_list`
- `note_clear`
- `plan_update`

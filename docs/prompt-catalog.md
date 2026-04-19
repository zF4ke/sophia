# Prompt Catalog

Runtime prompts remain external and are loaded through `PromptRegistry`.

## Active Prompt IDs

- `runtime/agent_loop`
- `runtime/stall_classifier`
- `system/personality_mixed_override`
- `system/personality_classic_override`

The stable code catalog is `src/shared/promptCatalog.ts`.

## Purpose

- `runtime/agent_loop`
  Unified system prompt for the native tool-calling loop. Defines identity, environment, research flow, evidence rules, tool calling rules, and voice.
- `runtime/stall_classifier`
  Lightweight binary classifier used by the stall guard to decide whether a finish-answer is a non-productive promise that should be rejected once.
- `system/personality_mixed_override`
  Mixed personality override injected when `/settings` → `Personalidade` is set to `mixed`.
- `system/personality_classic_override`
  Classic personality override injected when `/settings` → `Personalidade` is set to `classic`.


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
- `note_read`
- `note_update`
- `note_clear`
- `plan_update`

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
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`

# Agent Loop And Tool Contract

The agent loop is intentionally short and bounded.

## Flow

1. `RequestClassifier` decides between `direct_answer` and `discord_grounded`.
2. `AgentOrchestrator` either answers directly or starts a tool loop.
3. The loop asks the planner prompt for one next action at a time.
4. `DiscordToolService` executes the chosen tool.
5. The loop stops when evidence is good enough, clearly weak, or the step budget is exhausted.
6. The synthesis prompt builds the final grounded answer from compact evidence only.

## Available Tools

- `search_messages`
- `read_message_thread`
- `read_channel_summary`
- `list_relevant_channels`
- `crawl_channel_messages`
- `get_member_profile`
- `list_members`
- `get_guild_context`

## Important Rule

## Grounding Semantics

- `search_messages`, `read_message_thread`, and `read_channel_summary` provide message evidence.
- `get_guild_context`, `get_member_profile`, and `list_members` provide live evidence.
- `list_relevant_channels` and `crawl_channel_messages` are discovery tools and must not be treated as final evidence on their own.
- Live evidence can satisfy grounding for current-server facts even when there are no stored message hits.
- When local memory search misses, the agent may crawl readable unstored channels, ingest them, and rerun local search within the same bounded loop.
- `Fontes` should only appear when there are clickable message citations.

If a tool is changed, the runtime and prompt contract must change together. The most important synchronization points are:

- `src/shared/discordTools.ts`
- `src/discord/tools/DiscordToolService.ts`
- `src/agent/AgentOrchestrator.ts`
- `resources/prompts/tasks/plan_discord_search.md`

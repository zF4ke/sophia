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
- `get_member_profile`
- `list_members`
- `get_guild_context`

## Important Rule

If a tool is changed, the runtime and prompt contract must change together. The most important synchronization points are:

- `src/shared/discordTools.ts`
- `src/discord/tools/DiscordToolService.ts`
- `src/agent/AgentOrchestrator.ts`
- `resources/prompts/tasks/plan_discord_search.md`

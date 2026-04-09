# Agent Loop And Tool Contract

The agent loop is intentionally short and bounded.

## Flow

1. `RequestClassifier` decides between `direct_answer` and `discord_grounded`.
2. For grounded questions, `AgentOrchestrator` gathers explicit mentions, readable channel names, short-lived same-guild conversation context, and cache state.
3. The retrieval controller prompt chooses the current question intent, the next tool action, and the grounded answer mode.
4. `AgentOrchestrator` first checks for a fresh reusable grounded context in the same guild, preferring the current channel.
5. If reuse is not enough, `AgentOrchestrator` runs a bounded tool loop driven by repeated retrieval-controller decisions.
6. `DiscordToolService` executes the chosen tool, using TTL-based tool-result caching where appropriate.
7. The controller may choose channel discovery, live crawl, and local retry when local memory is weak.
8. Grounded answers use a graded outcome: `confident`, `best_effort`, or `insufficient`.
9. The synthesis prompt builds the final grounded answer from compact evidence only.

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
- For person-target questions about what someone said, live member/profile evidence is identity context only; message evidence is still required for a confident answer.
- When local memory search misses, the agent may crawl readable unstored channels, ingest them, and rerun local search within the same bounded loop.
- Exact readable channel-name matches such as `silksong` can be used as strong channel hints even without explicit `canal` wording.
- Live member lookups should retry bounded Discord rate limits rather than failing immediately.
- The retrieval controller is the main semantic decision-maker; deterministic logic remains only for explicit entities, permission/readability constraints, and loop bounds.
- Reusable grounded contexts are guild-wide by default, but same-channel matches should win when available.
- Tool-result caching uses TTL plus a short per-guild response-age limit in v1.
- Final answers should not append visible source lists or citation blocks.

If a tool is changed, the runtime and prompt contract must change together. The most important synchronization points are:

- `src/shared/discordTools.ts`
- `src/discord/tools/DiscordToolService.ts`
- `src/agent/AgentOrchestrator.ts`
- `resources/prompts/tasks/plan_discord_search.md`

# Prompt Catalog

Runtime prompts are stored on disk and loaded through `PromptRegistry`.

## Prompt IDs

- `system/base`
- `system/grounded`
- `tasks/classify_request`
- `tasks/route_discord_intent`
- `tasks/plan_discord_search`
- `tasks/judge_grounding_sufficiency`
- `tasks/synthesize_answer`
- `guards/insufficient_evidence`

The stable code catalog is `src/shared/promptCatalog.ts`.

## Tool-Aware Prompt

`resources/prompts/tasks/route_discord_intent.md` is the routing prompt for ambiguous grounded questions. It must stay aligned with the route decision shape in `src/shared/appTypes.ts`, including topic hints, channel-name hints, and reuse of short-lived prior person context.

`resources/prompts/tasks/plan_discord_search.md` is tool-aware and must stay aligned with:

- `search_messages`
- `read_message_thread`
- `read_channel_summary`
- `list_relevant_channels`
- `crawl_channel_messages`
- `get_member_profile`
- `list_members`
- `get_guild_context`

It also defines the grounding policy:

- message tools are for history and discussion evidence
- live metadata tools are authoritative for current-server facts
- `list_relevant_channels` and `crawl_channel_messages` are discovery-only
- the planner may pivot from local search to bounded live crawl, then rerun local search
- for person-target questions about what someone said, the planner should prefer author-scoped search using a topic hint instead of the whole natural-language question
- exact readable channel-name matches may be preserved as channel hints for person-target or channel-target searches
- the planner should finish once the available evidence already answers the question

`resources/prompts/tasks/judge_grounding_sufficiency.md` decides whether the currently retrieved evidence is enough to answer reliably. It must stay aligned with the grounding judgment shape in `src/shared/appTypes.ts`.

The judge is selective in runtime:

- obvious insufficient cases should skip it
- obvious sufficient cases should skip it
- ambiguous middle-ground cases should call it

## JSON Contracts

These prompts return strict JSON and must stay synchronized with `src/shared/appTypes.ts`:

- `tasks/classify_request`
- `tasks/route_discord_intent`
- `tasks/plan_discord_search`
- `tasks/judge_grounding_sufficiency`

If you change the output schema, update the prompt, types, runtime parser, tests, and this file together.

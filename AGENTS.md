# Sophia3 Agent Handbook

Sophia3 is an agentic Discord assistant with two data sources:

- Local message memory for Discord history and evidence retrieval.
- Live Discord API tools for transient metadata such as members, roles, channels, and guild state.

This file is the canonical engineering and operator handoff for the runtime. `README.md` is the product/setup overview. The deeper references live in `docs/`.

## Canonical Docs

- `docs/architecture.md`
- `docs/agent-loop.md`
- `docs/memory-indexing.md`
- `docs/prompt-catalog.md`
- `docs/commands-and-admin.md`
- `docs/cleanup-migration.md`
- `docs/ui.md`

## Runtime Flow

1. Classify the request as `direct_answer` or `discord_grounded`.
2. If grounded, apply deterministic routing for explicit Discord entities and clear server/member questions.
3. For ambiguous grounded questions, run the Discord routing prompt once to choose a target strategy, reusing short-lived same-guild conversation context for referential follow-ups when appropriate.
4. Run a bounded Discord tool loop from that route.
5. Treat stored-message evidence and live Discord metadata as separate grounding types.
6. Reuse a fresh grounded context from the same guild when possible, preferring the current channel over other guild channels.
7. Run a bounded Discord tool loop with cached tool results when reuse is not enough.
8. Use the grounding sufficiency judge only for ambiguous middle-ground cases; obvious sufficient and obvious insufficient cases stay heuristic.
9. Synthesize a concise answer only from useful evidence.
10. Keep citations internal for grounding and debug, but do not show source lists in the final user-facing answer.
11. If both grounding types are weak or the evidence judge says the evidence is insufficient, say that Sophia does not have enough Discord evidence to answer reliably.
12. If local memory misses, the agent may crawl readable channels, ingest them, and retry local retrieval within the same bounded loop.
13. Person-target follow-ups should preserve the resolved member identity, then search that person’s messages with topic or channel hints instead of falling back to member listing.

The main orchestration files are:

- `src/agent/RequestClassifier.ts`
- `src/agent/AgentOrchestrator.ts`
- `src/discord/tools/DiscordToolService.ts`
- `src/memory/DiscordMemoryService.ts`

## Stable Tool Contract

These tool names are part of the prompt contract and should stay stable unless you intentionally change the registry and prompts together:

- `search_messages`
- `read_message_thread`
- `read_channel_summary`
- `list_relevant_channels`
- `crawl_channel_messages`
- `get_member_profile`
- `list_members`
- `get_guild_context`

The stable code catalog is:

- `src/shared/discordTools.ts`

If you add, remove, or rename a tool, update all of the following in the same change:

- `src/shared/discordTools.ts`
- `src/discord/tools/DiscordToolService.ts`
- `src/agent/AgentOrchestrator.ts`
- `resources/prompts/tasks/plan_discord_search.md`
- `docs/prompt-catalog.md`
- Any affected tests, especially the documentation integrity test

If you change whether a tool is treated as message evidence, live evidence, or discovery-only, update:

- `src/shared/discordTools.ts`
- `src/agent/AgentOrchestrator.ts`
- `resources/prompts/tasks/plan_discord_search.md`
- `docs/agent-loop.md`
- `docs/prompt-catalog.md`

## Prompt Ownership

Runtime prompts live in `resources/prompts/` and should remain external to the code.

- `resources/prompts/system/base.md`
- `resources/prompts/system/grounded.md`
- `resources/prompts/tasks/classify_request.md`
- `resources/prompts/tasks/route_discord_intent.md`
- `resources/prompts/tasks/plan_discord_search.md`
- `resources/prompts/tasks/judge_grounding_sufficiency.md`
- `resources/prompts/tasks/synthesize_answer.md`
- `resources/prompts/guards/insufficient_evidence.md`

The stable prompt catalog is:

- `src/shared/promptCatalog.ts`

If you change the expected JSON shape for classification, routing, tool planning, or grounding sufficiency judgment, update:

- The prompt file
- The consuming TypeScript types in `src/shared/appTypes.ts`
- The calling code in `RequestClassifier` or `AgentOrchestrator`
- The docs in `docs/prompt-catalog.md`
- The related tests

## Models and Config

Editable runtime config lives in:

- `resources/models/model-profiles.json`
- `.env`
- `src/app/AppConfig.ts`

OpenRouter is the only provider surface in v1. Keep model profile edits in `resources/models/model-profiles.json` instead of hardcoding models in source.

## Memory and Indexing

Stored Discord messages are the source of truth for search and grounded answers.

- New messages are ingested from `src/discord/events/message/messageCreate.event.ts`
- Backfill and repair flows are exposed through `src/discord/commands/system/index.command.ts`
- Persistence and retrieval live under `src/memory/`
- Reusable grounded contexts, cached tool results, and short-lived conversation resolution context are stored in the same SQLite runtime database with TTL plus a short per-guild response-age limit

When changing retrieval behavior, update both code and docs:

- Retrieval code under `src/memory/`
- `docs/memory-indexing.md`
- Relevant prompts if retrieval behavior changes the agent’s expectations

## Commands

Core supported commands:

- `/ask`
- `/find`
- `/context`
- `/talk`
- `/nth`
- `/debug`
- `/index`
- `/access`
- `/cache`

The current interaction entrypoint is:

- `src/discord/events/interaction/interactionCreate.event.ts`

The current command registry/loader is:

- `src/platform/loaders/commandLoader.ts`

## How To Proceed Safely

When making behavior changes, prefer this sequence:

1. Update the smallest responsible runtime module.
2. Update prompts if the agent contract changed.
3. Update `AGENTS.md` and the relevant file in `docs/`.
4. Update or add tests.
5. Run `npm run check`.

Do not reintroduce large inline prompts, fake web-search claims, or raw transcript dumping into model context.

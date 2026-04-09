# Architecture Overview

Sophia3 is organized around an agentic Discord memory architecture:

- `src/app/` bootstraps the client and runtime.
- `src/agent/` classifies requests and runs the bounded tool loop.
- `src/ai/` owns model access.
- `src/discord/` holds commands, events, UI, live Discord access, and conversation helpers.
- `src/memory/` owns local persistence, indexing, and retrieval.
- `src/security/` owns admin/moderator state, command policies, and rate limiting.
- `src/platform/` owns loaders, HTTP plumbing, and storage helpers.
- `src/shared/` stores stable catalogs that resources/prompts/docs depend on.
- `resources/` stores tracked runtime assets.
- `storage/` stores mutable generated state.

Publicly stable catalogs:

- `src/shared/discordTools.ts`
- `src/shared/promptCatalog.ts`

The root bootstrap file is `src/index.ts`, but the actual runtime wiring now lives in `src/app/`.

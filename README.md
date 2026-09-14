<div align="center">

<img src="assets/logo.svg" alt="Sophia" width="76" />

# Sophia 5

*A conversational AI assistant for Discord that researches your server, works with files, remembers useful things, and acts with your permission.*

[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial-8b6dff)](LICENSE)
![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)
![discord.js](https://img.shields.io/badge/discord.js-v14-5865F2?logo=discord&logoColor=white)
![Node](https://img.shields.io/badge/node-22.19%2B-339933?logo=node.js&logoColor=white)
[![Documentation](https://img.shields.io/badge/docs-Sophia%205-8b6dff)](https://zf4ke.github.io/sophia/)

**[Website and handbook](https://zf4ke.github.io/sophia/)** &nbsp; **[How it works](docs/how-sophia-works.md)** &nbsp; **[Architecture](docs/architecture.md)** &nbsp; **[Testing](docs/testing.md)**

</div>

---

Sophia is a Discord assistant with one identity, durable tasks, automatic memory and a model-led tool loop. It can research a question, work with files, accept corrections while working, and carry authorized tasks through to completion. See [the implementation plan](docs/v5-plan.md) and [validation record](docs/live-acceptance.md) for the current release checks.

The [Sophia handbook](https://zf4ke.github.io/sophia/) is the website and documentation destination. This branch replaces the old landing page with a Starlight site; publishing happens through the Pages workflow after merge. Preview the new site locally with `npm run docs:build` and `npm run docs:preview` after installing its dependencies in `website/`.

Sophia can research Discord history and the web, work with files in an isolated container, inspect images and video frames, transcribe audio with a compatible configured model, maintain memories and procedural skills, and run scheduled follow-ups. Ordinary conversation uses the same runtime as longer work.

## Run locally

Use Node.js 22.19 or newer for the bot and docs together; the bot toolchain supports Node 20.19 or Node 22.12 and newer. Install dependencies with `npm ci`, copy `.env.example` to `.env`, and configure the Discord token, OpenRouter key and `OPENCODE_API_KEY`. Set `BOOTSTRAP_ADMIN_IDS` to the operator's verified Discord user ID for initial administration. Secrets belong in `.env`; runtime settings belong in `storage/settings.json`.

Muse Spark 1.3 Free through OpenCode Zen is the fresh-install default. Sophia uses its own runtime with the Responses API and task session metadata. OpenRouter profiles and local compatible servers remain selectable in `/settings`. Existing installations retain their saved selection. Embeddings still use OpenRouter. See the [model guide](website/src/content/docs/setup/models.md) for provider availability and data-use terms.

Run `npm start`, or `npm run dev` for development. The process registers its commands through the normal startup flow. Only one process may own a storage directory.

New installations enable no guilds and disable DMs. An operator enables a guild with `/access enable_here:true` and grants a user access with `/access user:@person level:write mode:ask`. Global grants still require enabled locations. `/access enable_dms:true` enables DMs for authorized users. Reads run without approval; changes follow the granted tier and Ask or Auto mode.

The workspace requires a functioning Docker engine and the local image built with `npm run sandbox:build`. It has no host-execution fallback. Configure model profiles in `resources/models/model-profiles.json` and select one in `/settings`. Declared model input modalities control image and audio support. A profile can use an explicit compatible endpoint; otherwise model requests use OpenRouter.

## Use Sophia

Use `npm run state -- backup <new-directory>` with Sophia stopped to archive durable state. Restore into empty storage with `npm run state -- restore <archive-directory> <empty-storage-directory>`. Credentials and reconstructible indexes are excluded. See [cleanup and migration](docs/cleanup-migration.md).

- `/talk`, a mention, a reply, or an enabled DM starts a conversation. Attachments can be supplied without text.
- Mention Sophia or reply to steer or stop work in plain language. Describe the task; she can find it without you copying an ID. `/steer` and `/stop` remain optional controls.
- Ask to continue or revise earlier work by description. Sophia can reopen its saved notes, files and previous answer. `/tasks` remains available to inspect plans, evidence, usage and files.
- `/schedules` manages your scheduled work. Conditional checks can stay quiet when nothing needs attention.
- `/settings` controls the selected model, voice and runtime configuration. `/access` manages availability and authority.
- `/costs` reports your usage in the channel by default; `ephemeral:true` keeps it private. Operators have a private installation-wide Custos view in `/settings`. Missing prices and tokens remain explicit.
- Research uses locally indexed messages first and fetches missing history when needed. `/index` remains an operator inspection and repair command.

Tasks have no default total call, duration or cost cap. Cancellation, repeated-call protection, context capacity and bounded individual operations remain. An optional explicit tool-call limit applies to the whole task, including continuation legs. Usage estimates describe configured prices, not provider billing receipts.

## Storage and recovery

`tasks.sqlite` stores work, receipts, approvals, collections and files. `knowledge.sqlite` stores identity, memories and dreaming jobs. `products.sqlite` stores cards and skills. The operational retrieval database is rebuildable; the durable stores and settings are not disposable.

After a restart, interrupted work pauses. External actions are not replayed automatically. Unknown action outcomes must be checked before a task resumes. Settings resets preserve access rules; index resets preserve durable work.

## Engineering

Run `npm run doctor` to check local prerequisites without calling a model or sending a message. For a second PC, follow the [installation guide](website/src/content/docs/setup/installation.md) and [transfer guide](website/src/content/docs/setup/another-pc.md). Build the Docker image on that PC and recreate `.env` separately. Never sync a live SQLite directory between running installations.

Run `npm run check` for type checking and deterministic tests. `npm run test:live` runs opt-in integration tests; consult [testing](docs/testing.md) for prerequisites. Real Discord behavior, provider behavior and container isolation need integration verification in addition to mocks.

Start with [architecture](docs/architecture.md), [the agent loop](docs/agent-loop.md), [access policy](docs/access-policy.md), [task lifecycle](docs/task-lifecycle.md), [workspace and media](docs/workspace-and-media.md), [memory](docs/memory-indexing.md), [skills](docs/skills.md), and [scheduling](docs/scheduling.md). Tool contracts and engineering rules live in [AGENTS.md](AGENTS.md).

## Launch the bot and website

From the repository root, install dependencies once on each PC:

```sh
npm ci
npm --prefix website ci
```

With `.env` configured and Docker running, build the sandbox once, then start Sophia:

```sh
npm run sandbox:build
npm run doctor
npm start
```

In a separate terminal, start the documentation website:

```sh
npm run docs:dev
```

Open the localhost address printed by Astro, normally `http://127.0.0.1:4321/`. To test the static site with its generated search index, stop the docs dev server and run:

```sh
npm run docs:build
npm run docs:preview
```

The public website target is **https://zf4ke.github.io/sophia/**. The Pages workflow builds the `/sophia/` base path and publishes `website/dist` when these changes reach `master`, or when that workflow is run manually. Repository Pages should use **GitHub Actions** as its source. The website contains static documentation; the bot runs separately on your PC. See the [website guide](website/src/content/docs/setup/website.md).

## License

Sophia is distributed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

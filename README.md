# Sophia3

An agentic Discord assistant backed by local Discord memory, OpenRouter models, and bounded retrieval tools.

For engineering/runtime guidance, use `AGENTS.md`. For deeper reference docs, use `docs/`.

## Features

- **Agentic Answers**: Sophia decides when to answer directly and when to retrieve Discord evidence.
- **Local Discord Memory**: Messages are stored locally in SQLite and searched with lexical plus embedding retrieval.
- **Grounded Search**: Evidence search works across readable channels and returns clickable citations.
- **Live Discord Tools**: Member and guild metadata are fetched live instead of being blindly persisted.
- **Modern Interactions**: Current `discord.js` components with buttons and select menus for evidence browsing.

## Tech Stack

- TypeScript
- Discord.js
- OpenRouter via the `openai` SDK
- SQLite (`better-sqlite3`)
- Express
- Vitest

## Project Structure

```
src/
├── app/          # Bootstrap, runtime config, shared path registry
├── agent/        # Classification, orchestration, prompt loading
├── ai/           # Model gateway
├── discord/      # Commands, events, UI, live Discord access, conversation helpers
├── memory/       # Persistence, indexing, retrieval, ranking
├── platform/     # Loaders, HTTP app, storage helpers
├── security/     # Admin/moderator state, policies, rate limits
└── shared/       # Cross-domain contracts and stable catalogs
resources/        # Repo-tracked runtime assets such as prompts and model profiles
storage/          # Mutable local runtime state
tests/            # Test tree mirroring runtime domains
```

## Services

### AI and Agent Services
- `ModelGateway`: OpenRouter-backed model access
- `PromptRegistry`: Filesystem-based prompt loading
- `AgentOrchestrator`: Bounded tool loop and grounded answering
- `RequestClassifier`: Direct-answer vs Discord-grounded routing

### Core Services
- `DiscordMemoryService`: Message ingestion, indexing, and retrieval facade
- `DiscordBackfillService`: Historical indexing for channels and categories
- `DiscordToolService`: Internal Discord tools for the agent
- `ConversationService`: Message grouping and threading
- `SecurityService`: Access control and permissions facade
- `UIService`: User interface facade
- `AppPaths`: Central path registry for resources, storage, commands, and events

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
```env
DISCORD_TOKEN=your_discord_token
OPENROUTER_API_KEY=your_openrouter_api_key
MODEL_PROFILE=balanced
```
Tracked runtime assets live under `resources/`. Mutable state is written under `storage/`.

3. Start development server:
```bash
npm run dev
```

4. Run the bot:
```bash
npm start
```

## Commands

### Core Commands
- `/ask`: Ask Sophia a question
- `/find`: Search stored Discord evidence
- `/context`: Ask for a context-aware answer
- `/talk`: Talk to Sophia directly
- `/getmessage`: Retrieve the Nth indexed historical message
- `/index`: Backfill, inspect, repair, or clear the local memory index

### System Commands
- `/access`: Manage user permissions
- `/cache`: Show local memory statistics

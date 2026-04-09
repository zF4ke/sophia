# Sophia3

An agentic Discord assistant backed by local Discord memory, OpenRouter models, and bounded retrieval tools.

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
├── api/          # API endpoints
├── commands/     # Bot commands
├── events/       # Discord event handlers
├── handlers/     # Command and event handlers
├── services/     # Core services
│   ├── ai/      # AI-related services
│   └── ui/      # UI components
├── types/        # TypeScript type definitions
└── utils/        # Utility functions
```

## Services

### AI and Agent Services
- `ModelGateway`: OpenRouter-backed model access
- `PromptRegistry`: Filesystem-based prompt loading
- `AgentOrchestrator`: Bounded tool loop and grounded answering
- `RequestClassifier`: Direct-answer vs Discord-grounded routing

### Core Services
- `DiscordMemoryService`: Message ingestion, indexing, and retrieval
- `DiscordBackfillService`: Historical indexing for channels and categories
- `DiscordToolService`: Internal Discord tools for the agent
- `ConversationService`: Message grouping and threading
- `SecurityService`: Access control and permissions
- `FileSystemService`: Data persistence and cache management
- `UIService`: User interface components

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

3. Start development server:
```bash
npm run dev
```

4. Build and run production:
```bash
npm run build
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

# Sophia

A conversational Discord assistant that can search server history, manage channels, and take actions — all through natural language.

Sophia turns your Discord server into a searchable, manageable workspace. Ask her questions about what happened in any channel, who said what, or have her create channels, manage roles, and clean up messages — with a built-in approval system that keeps admins in control.

## Features

**Conversational AI** — Talk to Sophia naturally through `/talk`, mentions, or replies. She maintains conversation continuity across reply chains and threads.

**Deep Server Search** — Search through your entire Discord history with intelligent retrieval. Sophia indexes messages locally for fast lookups and automatically fetches live history when needed.

**21 Built-in Tools** — From retrieving messages and resolving members to creating channels, managing roles, and sending messages. The model decides which tools to use based on your request.

**Admin Approval System** — Write actions (create channel, send message, manage roles) require admin approval. Destructive actions (delete channel, clear messages) require approval *plus* confirmation. Multiple destructive actions are batched into a single approval card grouped by Discord category.

**Auto-Approve Mode** — Optionally skip approval for non-destructive write actions when you trust the model.

**Animated Activity Indicators** — Sophia reacts with a cycling emoji sequence while thinking, so you always know she's working.

**Configurable Runtime** — Tune tool-call limits, latency budgets, retrieval depth, approval timeouts, and more through an interactive `/settings` panel. Switch between model profiles on the fly.

**Role-Based Access** — Admin and moderator tiers with rate limiting. Admins get full control; moderators get scoped permissions.

## Quick Start

### Prerequisites

- Node.js 18+
- A Discord bot token
- An [OpenRouter](https://openrouter.ai/) API key

### Install

```bash
git clone <repo-url>
cd sophia
npm install
```

### Configure

Create a `.env` file:

```env
DISCORD_TOKEN=your-discord-bot-token
OPENROUTER_API_KEY=your-openrouter-api-key
```

### Run

```bash
npm start
```

For development with auto-reload:

```bash
npm run dev
```

## Commands

### Conversation

| Command | Description |
|---------|-------------|
| `/talk` | Start a conversation with Sophia |
| `@Sophia` | Mention Sophia in any message |
| *Reply* | Reply to any Sophia message to continue the conversation |

### Search

| Command | Description |
|---------|-------------|
| `/find` | Targeted retrieval with topic, channel, and author filters |
| `/nth` | Read the N-th historical message from an indexed channel |

### Admin

| Command | Description |
|---------|-------------|
| `/settings` | Interactive panel to configure runtime parameters and model profile |
| `/index` | Manage message indexing — backfill channels/categories, check status, repair |
| `/debug` | Toggle debug mode or view model output logs |
| `/access` | Manage admin and moderator access |
| `/ping` | Health check with latency info |

## Tools

Sophia has **21 tools** organized by capability:

| Category | Tools |
|----------|-------|
| **Retrieval** | `retrieve_messages` |
| **Discovery** | `list_guild_structure`, `get_guild_context`, `resolve_channel_targets` |
| **Members** | `resolve_member_identity`, `get_member_profile`, `list_members`, `get_role_info` |
| **Threads** | `list_threads`, `read_thread_messages` |
| **Utilities** | `measure_text_length`, `evaluate_math` |
| **Write** | `create_channel`, `create_category`, `create_thread`, `move_channel`, `manage_member_roles`, `send_message` |
| **Destructive** | `clear_messages`, `delete_channel` |

Write tools require admin approval. Destructive tools require approval + confirmation dialog.

## Model Profiles

Three pre-configured profiles, switchable via `/settings`:

| Profile | Model | Context Window |
|---------|-------|----------------|
| **fast** (default) | Gemini 3.1 Flash Lite | 1M tokens |
| **smarter** | MiniMax M2.7 | 190K tokens |
| **alt** | DeepSeek V3.2 | 160K tokens |

All models are accessed through OpenRouter. No models are hardcoded — profiles are defined in `resources/models/model-profiles.json`.

## How It Works

Sophia uses a **while-loop with native function calling**. One unified system prompt gives the model the question, conversation context, and available tools. The model calls tools iteratively and calls `finish` when it has an answer.

There is no separate planner or step selector. The model handles all decisions inside the tool-calling loop. The runtime enforces budgets, guardrails, and approval gates.

**Retrieval pipeline:**
1. Search local indexed Discord messages
2. If evidence is weak, refresh from live Discord history
3. Ingest new messages into the local cache
4. Retry retrieval on the enriched cache

**Conversation continuity** follows reply chains across threads and channels, so `/talk`, mentions, and replies all feel like one continuous conversation.

## Configuration

All runtime tuning is done through `/settings` or `storage/settings.json`:

- **Context retention** — recent turns, channel messages, prior evidence slice
- **Retrieval** — history page size, context window, crawl limits
- **Loop guardrails** — max tool calls (2–30), latency budget (10s–5m), repeated call guard
- **Approval** — timeout duration (30s–5m), auto-approve writes toggle
- **Model profile** — switch between fast/smarter/alt

## Scripts

| Script | Purpose |
|--------|---------|
| `npm start` | Run the bot |
| `npm run dev` | Development mode with auto-reload |
| `npm run check` | Typecheck + run all deterministic tests |
| `npm run test` | Fast deterministic test suite |
| `npm run test:live` | Real-model tests (requires API key) |
| `npm run clean` | Reset all storage to defaults |

## Documentation

| Doc | Topic |
|-----|-------|
| [Architecture](docs/architecture.md) | System blocks, runtime flow, storage |
| [Agent Loop](docs/agent-loop.md) | While-loop mechanics, tool model, guardrails |
| [How Sophia Works](docs/how-sophia-works.md) | End-to-end flow, retrieval deep dive, capability composition |
| [Commands & Admin](docs/commands-and-admin.md) | All commands, approval system, access control |
| [Memory & Indexing](docs/memory-indexing.md) | Local cache, retrieval strategy, `/index` commands |
| [Prompt Catalog](docs/prompt-catalog.md) | Active prompts and stable capability names |
| [Feature Stories](docs/feature-user-stories.md) | Concrete acceptance stories for all tools and compositions |
| [Testing](docs/testing.md) | Test strategy, deterministic + live suites, manual tests |
| [UI Guidelines](docs/ui.md) | Response formatting, approval cards, activity indicators |
| [Cleanup & Migration](docs/cleanup-migration.md) | What was removed and why |

## Tech Stack

- **Runtime:** Node.js + TypeScript
- **Discord:** discord.js v14
- **AI:** OpenRouter (OpenAI SDK compatible)
- **Storage:** libSQL (local SQLite)
- **Validation:** Zod
- **Testing:** Vitest

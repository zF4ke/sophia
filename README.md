# Sophia3

A powerful Discord bot leveraging Google's Gemini AI for intelligent message searching, conversation analysis, and context-aware responses.

## Key Features

- **AI-Powered Search**: Natural language search using Gemini API
- **Smart Conversation Analysis**: Groups and analyzes message context
- **Interactive UI**: Rich embeds with pagination and interactive controls
- **Context-Aware Responses**: AI responses based on channel conversation history
- **Security Controls**: Command-level access management
- **Ephemeral Responses**: Optional private results

## Available Commands

### Core Commands
- `/search [topic] [channel] [limit] [include_bots] [ephemeral]` - Search conversations
- `/context [prompt] [channel] [limit] [include_bots] [ephemeral]` - Get AI responses with context
- `/getmessage [channel] [number] [ephemeral]` - Retrieve specific messages

### System Commands
- `/access` - Manage command access
- `/ping` - Check bot status
- `/cache` - Manage message cache

## Quick Start

1. **Prerequisites**
   - Node.js v18+
   - Discord.js v14+
   - Google Cloud account with Gemini API access
   - Discord Bot Token

2. **Installation**
```bash
# Clone repository
git clone https://github.com/yourusername/Sophia3.git
cd Sophia3

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your tokens
```

3. **Start the Bot**
```bash
# Development
npm run dev

# Production
npm start
```

## Architecture

```
src/
├── commands/    # Discord slash commands
├── events/      # Discord event handlers
├── services/    # Core business logic
│   ├── ai/      # AI services (Gemini integration)
│   └── ui/      # UI components
└── utils/       # Utilities and constants
```

## Documentation

- [API Documentation](./docs/API.md) - Technical details
- [Getting Started Guide](./docs/guides/GettingStarted.md) - Setup tutorial
- [Examples](./docs/guides/Examples.md) - Code examples

## Built With

- [Discord.js](https://discord.js.org/) - Discord API framework
- [Google Generative AI (Gemini)](https://cloud.google.com/ai/generative-ai) - AI capabilities
- TypeScript - Type safety and modern JavaScript features

## Required Permissions

Bot requires the following Discord permissions:
- Read Messages/View Channels
- Send Messages
- Embed Links
- Read Message History
- Use Application Commands

Permission Integer: `274878286912`
# Sophia3

A powerful Discord bot leveraging Google's Gemini AI for intelligent message searching, conversation analysis, and context-aware responses.

## Features

- **AI-Powered Search**: Find relevant conversations using semantic search and context analysis
- **Conversation Analysis**: Analyze chat context and relevance using Google's Gemini AI
- **Message Management**: Efficient caching and smart message grouping
- **Interactive UI**: Rich embeds with pagination and interactive controls
- **Security**: Role-based access control and command restrictions

## Tech Stack

- TypeScript
- Discord.js
- Google Gemini AI
- Express (API Server)
- Node.js

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

### AI Services
- `AIService`: Primary interface for AI operations
- `ConversationAnalysisService`: Analyzes chat relevance and context
- `ContextManagementService`: Manages conversation context
- `ResponseGenerationService`: Generates AI responses
- `TextProcessingService`: Text preprocessing and analysis

### Core Services
- `ConversationService`: Message grouping and threading
- `MessageService`: Message operations and caching
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
GOOGLE_AI_KEY=your_gemini_api_key
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
- `/search`: Search through conversations
- `/context`: Get AI-powered responses
- `/getmessage`: Retrieve specific messages

### System Commands
- `/access`: Manage user permissions
- `/cache`: Manage message cache
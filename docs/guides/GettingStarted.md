[Back to Index](../API.md)

# Getting Started with Sophia3

Sophia3 is a Discord bot that provides advanced conversation analysis and AI-powered search capabilities using Google's Generative AI (Gemini). This guide will help you get started with setting up and using Sophia3 in your Discord server.

## Prerequisites

Before you begin, make sure you have the following:

1. **Node.js** (v16.x or higher) and **npm** (v7.x or higher) installed
2. **Discord Bot Token** - Create a bot in the [Discord Developer Portal](https://discord.com/developers/applications)
3. **Google API Key** with access to the Generative AI (Gemini) API

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Sophia3.git
cd Sophia3
```

2. **Install dependencies**

```bash
npm install
```

3. **Set up environment variables**

Create a `.env` file in the root directory with the following variables:

```env
CLIENT_TOKEN=your_discord_bot_token
GOOGLE_API_KEY=your_google_api_key
```

## Core Concepts

### Service Architecture

Sophia3 uses a layered service architecture:

1. **Core Services** - Base functionality and utilities
2. **UI Services** - Display and interaction handling
   - Base UIService with common UI functionality
   - Specialized services (like SearchUIService) for specific features
3. **AI Services** - AI-powered analysis and processing

### UI System

The UI system follows a hierarchical structure:

```
UIService (Base)
└── SearchUIService
    └── Future specialized UI services...
```

#### Key Features:
- Standardized pagination
- Interactive navigation
- Error handling
- Message grouping
- Embed formatting

## Running the Bot

To start the bot in development mode with hot reloading:

```bash
npm run dev
```

For production:

```bash
npm start
```

## Bot Permissions

Ensure your bot has the following permissions in Discord:

- **Read Messages/View Channels** - To read messages in channels
- **Send Messages** - To reply with search results
- **Embed Links** - To send rich embeds with search results
- **Read Message History** - To fetch message history for analysis
- **Use Application Commands** - To use slash commands

These permissions correspond to the following permission integer: `274878286912`

## Available Commands

Sophia3 provides the following slash commands:

### `/search`
Search for conversations in a channel about a specific topic.

**Parameters:**
- `channel`: The channel to search in
- `topic`: The topic to search for
- `limit` (optional): Maximum number of messages to fetch (default: 2000)
- `include_bots` (optional): Whether to include bot messages (default: false)
- `ephemeral` (optional): Whether to show results only to you (default: false)

**Example:**
```
/search channel:#general topic:project-planning limit:1000
```

### `/context`
Use channel messages as context for an AI query.

**Parameters:**
- `channel`: The channel to use as context
- `prompt`: Your question or instruction for the AI
- `limit` (optional): Maximum number of messages to fetch (default: 1000)
- `include_bots` (optional): Whether to include bot messages (default: false)
- `ephemeral` (optional): Whether to show response only to you (default: false)

**Example:**
```
/context channel:#project-team prompt:What was decided about the deadline? limit:500
```

### `/getmessage`
Retrieve a specific message by its position in a channel.

**Parameters:**
- `channel`: The channel containing the message
- `number`: The message number counting from the start of the channel
- `ephemeral` (optional): Whether to show result only to you (default: false)

**Example:**
```
/getmessage channel:#announcements number:50
```

## Basic Usage

### 1. Search Implementation

```typescript
// Search command setup
const searchResults = await AIService.analyzeConversations(conversations, topic);
await SearchUIService.displaySearchResults(interaction, searchResults, topic, channel.name);
```

### 2. Creating Custom UI Services

Extend the base UIService for custom implementations:

```typescript
export class CustomUIService extends UIService {
    public static async displayCustomContent(
        interaction: ChatInputCommandInteraction,
        content: any
    ): Promise<void> {
        // Your custom display logic
    }
}
```

## Configuration

### Message Display
- Default items per page: 5
- Collector timeout: 5 minutes
- Embed field limits: Following Discord's restrictions

### Navigation
- Primary/Secondary button styles
- Automatic button state management
- Permission-based interaction control

## Best Practices

1. **Error Handling**
   - Always handle interaction timeouts
   - Implement proper error recovery
   - Use built-in error handlers

2. **Performance**
   - Use pagination for large datasets
   - Group messages efficiently
   - Handle message chunking appropriately

3. **User Experience**
   - Use ephemeral responses when appropriate
   - Provide clear navigation options
   - Include relevant metadata in displays

## Next Steps

Now that you have Sophia3 running, check out the [Integration Examples](./Examples.md) for more advanced usage scenarios and code examples.

For detailed API documentation, refer to the [API Documentation](../API.md).

1. Review the [Examples](./Examples.md) for detailed usage scenarios
2. Check the [API Documentation](../API.md) for complete reference
3. Explore service-specific documentation in the services folder

## Troubleshooting

### Common Issues

1. **Rate Limiting**
   - If fetching large numbers of messages, Discord may rate-limit the bot
   - Solution: Use smaller limits or add delays between commands

2. **AI API Issues**
   - If AI responses fail, the bot will use fallback keyword search
   - Check your Google API key permissions and quotas

3. **Permission Errors**
   - Ensure the bot has proper permissions in the channels you're searching
   - Make sure the bot can see the channels mentioned in commands

### Getting Help

If you encounter any issues:

1. Check the console logs for detailed error messages
2. Verify your environment variables are set correctly
3. Ensure you have the latest version of the bot
4. Consult the [API Documentation](../API.md) for detailed reference
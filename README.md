# Sophia3

A Discord bot that helps find and display relevant message history using the Gemini API. It can group related messages into conversations and evaluate their relevance to search topics.

## Features

- **Message Search**: Find related messages using topic keywords or natural language queries
- **Message Organization**: Groups messages into conversations based on time and context
- **Interactive Display**: Navigate through search results with pagination
- **Relevance Scoring**: Uses Gemini API to help determine message relevance
- **Private Responses**: Option to show results only to the command user

## Commands

- `/search [topic]` - Find messages related to a topic
- `/context [message]` - Show messages around a specific message
- `/getmessage [message_id]` - Get a specific message and its context

## Project Structure

Three main layers:

1. **Commands**: Discord slash commands
2. **Services**: Core functionality
3. **AI Integration**: Gemini API integration

Main components:
- `AIService`: Handles Gemini API interactions
- `ConversationService`: Groups related messages
- `UIService`: Handles Discord message display
- `ContextManagementService`: Manages message context

## Documentation

See the [docs](./docs/API.md) folder for detailed technical documentation.

## Requirements

- Node.js v18+
- Discord.js v14+
- Google Cloud Platform account
- Discord Bot Token

## Built With

- [Discord.js](https://discord.js.org/)
- [Google Generative AI (Gemini)](https://cloud.google.com/ai/generative-ai)
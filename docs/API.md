# Sophia3 API Documentation

## Overview

Sophia3 is a Discord bot that uses the Gemini API to help find and organize relevant messages from a channel's history. It provides tools for message searching, conversation grouping, and interactive result display.

## Table of Contents

- **Core Services**
  - [MessageService](./services/MessageService.md) - Message fetching and filtering
  - [ConversationService](./services/ConversationService.md) - Message grouping logic
  - [UIService](./services/UIService.md) - Discord message display components
    - [ConversationUIService](./services/ui/ConversationUIService.md) - Conversation display controls

- **AI Integration**
  - [AIService](./services/AIService.md) - Main Gemini API interface
  - [AIBaseService](./services/AIBaseService.md) - Common AI functionality
  - [ContextManagementService](./services/ContextManagementService.md) - Message context handling
  - [ConversationAnalysisService](./services/ConversationAnalysisService.md) - Relevance evaluation
  - [ResponseGenerationService](./services/ResponseGenerationService.md) - AI response handling
  - [TextProcessingService](./services/TextProcessingService.md) - Text preparation utilities

- **Types & Constants**
  - [Types](./types/Types.md) - TypeScript type definitions
  - [Constants](./utils/Constants.md) - Configuration constants

- **Guides**
  - [Getting Started](./guides/GettingStarted.md) - Basic setup and usage
  - [Integration Examples](./guides/Examples.md) - Code examples

## Project Structure

The code is organized in three layers:

1. **Commands**: Discord slash command handlers
2. **Services**: Core business logic
3. **AI Integration**: Gemini API interaction

## Basic Usage

```typescript
// Example: Search for messages about a topic
const messages = await MessageService.fetchMessages(channel, limit, interaction);
const filteredMessages = MessageService.filterCommandMessages(messages, interaction);
const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
const validConversations = ConversationService.filterValidConversations(conversations, includeBots);
const relevantConversations = await AIService.analyzeConversations(validConversations, topic, interaction);
await ConversationUIService.displayConversations(interaction, relevantConversations, topic, channel.name, ephemeral);
```

For implementation details and more examples, see the individual service documentation.
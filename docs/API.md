# Sophia3 API Documentation

## Overview

Sophia3 is a Discord bot that provides advanced conversation analysis and AI-powered search capabilities using Google's Generative AI (Gemini). This documentation covers the core services and APIs available for developers.

## Table of Contents

- **Core Services**
  - [MessageService](./services/MessageService.md) - Handles Discord message fetching and filtering
  - [ConversationService](./services/ConversationService.md) - Groups messages into meaningful conversations
  - [UIService](./services/UIService.md) - Manages Discord UI elements and pagination
    - [ConversationUIService](./services/ui/ConversationUIService.md) - Conversation display and navigation

- **AI System**
  - [AIService](./services/AIService.md) - Main AI service entry point
  - [AIBaseService](./services/AIBaseService.md) - Base service for AI functionality
  - [ContextManagementService](./services/ContextManagementService.md) - Manages conversation contexts
  - [ConversationAnalysisService](./services/ConversationAnalysisService.md) - Analyzes conversations for relevance
  - [ResponseGenerationService](./services/ResponseGenerationService.md) - Generates AI responses
  - [TextProcessingService](./services/TextProcessingService.md) - Processes text for AI operations

- **Types & Constants**
  - [Types](./types/Types.md) - Core type definitions
  - [Constants](./utils/Constants.md) - Constants and utilities

- **Guides**
  - [Getting Started](./guides/GettingStarted.md) - How to get started with Sophia3
  - [Integration Examples](./guides/Examples.md) - Example usage scenarios

## Architecture

Sophia3 employs a layered architecture:

1. **Command Layer** - User-facing Discord commands
2. **Service Layer** - Core business logic for message handling and AI interaction
3. **AI Layer** - Specialized AI services for different aspects of AI functionality

## Quick Start

```typescript
// Example of basic conversation search with AI analysis
const messages = await MessageService.fetchMessages(channel, limit, interaction);
const filteredMessages = MessageService.filterCommandMessages(messages, interaction);
const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
const validConversations = ConversationService.filterValidConversations(conversations, includeBots);
const relevantConversations = await AIService.analyzeConversations(validConversations, topic, interaction);
await ConversationUIService.displayConversations(interaction, relevantConversations, topic, channel.name, ephemeral);
```

For more examples and detailed documentation for each service, please refer to the specific service documentation.
[Back to Index](../API.md)

# Types

This document describes the core type definitions used throughout the Sophia3 application.

## Conversation Types

### ConversationWithContext

```typescript
interface ConversationWithContext {
    messages: Message[];     // Array of Discord messages in the conversation
    relevanceScore: number;  // AI-assigned relevance score (0-10)
}
```

Represents a conversation with its relevance score as determined by AI analysis. Used extensively in search results and AI processing.

### MessageGroup

```typescript
interface MessageGroup {
    author: string;      // Author's username
    content: string[];   // Array of message contents
    timestamp: number;   // Timestamp of first message
}
```

Used for grouping messages by author for display purposes, especially in the UIService. Simplifies UI display by combining consecutive messages from the same author.

### AIAnalysisResult

```typescript
interface AIAnalysisResult {
    isRelevant: boolean;     // Whether the conversation is relevant
    relevanceScore: number;  // Relevance score (0-10)
}
```

Represents the result of an AI analysis on a conversation's relevance. The `relevanceScore` is a numeric value from 0-10 indicating how relevant the conversation is to a specific topic.

## Discord Client Type

### TDiscordClient

```typescript
type TDiscordClient = Client & {
    commands: Collection<string, any>;
};
```

Extends the Discord.js Client type with a commands collection to store and manage slash commands.

## Usage Examples

### ConversationWithContext Example

```typescript
// Creating a ConversationWithContext object
const relevantConversation: ConversationWithContext = {
    messages: messageArray, // Array of Discord Message objects
    relevanceScore: 8       // High relevance score on a scale of 0-10
};

// Accessing data
const messageCount = relevantConversation.messages.length;
const firstMessageAuthor = relevantConversation.messages[0].author.username;
const relevance = relevantConversation.relevanceScore;
```

### MessageGroup Example

```typescript
// Creating a MessageGroup for UI display
const userMessages: MessageGroup = {
    author: "ExampleUser",
    content: [
        "First message content",
        "Second message content from same user",
        "Third follow-up message"
    ],
    timestamp: 1625097600000 // Unix timestamp
};

// Using in a UI display context
messageGroups.forEach(group => {
    renderAuthorHeader(group.author, new Date(group.timestamp));
    group.content.forEach(msg => renderMessageContent(msg));
});
```

### AIAnalysisResult Example

```typescript
// Result from AI conversation analysis
const analysisResult: AIAnalysisResult = {
    isRelevant: true,
    relevanceScore: 7.5
};

// Using the result for filtering
if (analysisResult.isRelevant && analysisResult.relevanceScore >= 5) {
    includeConversationInResults(conversation);
}
```
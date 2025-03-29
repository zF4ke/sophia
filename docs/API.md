# Sophia3 API Documentation

## Services

### MessageService
Handles Discord message fetching and filtering operations.

#### Methods:
- `fetchMessages(channel: TextChannel, limit: number, interaction: ChatInputCommandInteraction): Promise<Message[]>`
  - Fetches messages from a Discord channel with progress updates
  - Supports up to 20,000 messages
  - Handles rate limiting automatically
  - Parameters:
    - `channel`: The Discord text channel to fetch from
    - `limit`: Maximum number of messages to fetch
    - `interaction`: Discord interaction for progress updates
  - Returns: Array of Discord messages

- `filterCommandMessages(messages: Message[], interaction: ChatInputCommandInteraction): Message[]`
  - Filters out command messages and recent messages from the command author
  - Parameters:
    - `messages`: Array of messages to filter
    - `interaction`: Current command interaction
  - Returns: Filtered array of messages

### ConversationService
Groups messages into meaningful conversations based on time gaps and relationships.

#### Methods:
- `groupMessagesByConversation(messages: Message[]): Message[][]`
  - Groups messages into conversations based on time gaps and relationships
  - Uses smart timing:
    - 5 minutes for regular messages
    - 8 minutes for same author
    - 10 minutes for direct replies
  - Parameters:
    - `messages`: Array of messages to group
  - Returns: Array of conversation groups (each group is an array of messages)

- `filterValidConversations(conversations: Message[][]): Message[][]`
  - Filters out invalid conversations (too short, bot-only, etc.)
  - Parameters:
    - `conversations`: Array of conversation groups
  - Returns: Filtered array of valid conversations

### AIService
Handles AI analysis of conversations using Google's Generative AI (Gemini).

#### Methods:
- `analyzeConversations(conversations: Message[][], topic: string, interaction: ChatInputCommandInteraction): Promise<ConversationWithContext[]>`
  - Analyzes conversations for relevance to a topic using Gemini AI
  - Features:
    - Batch processing to optimize API usage
    - Dynamic batch sizing based on conversation count
    - Progress updates during analysis
  - Parameters:
    - `conversations`: Array of conversation groups to analyze
    - `topic`: The search topic to analyze against
    - `interaction`: Discord interaction for progress updates
  - Returns: Array of relevant conversations with context

- `fallbackKeywordSearch(conversations: Message[][], topic: string): ConversationWithContext[]`
  - Provides basic keyword search when AI analysis fails
  - Parameters:
    - `conversations`: Array of conversation groups
    - `topic`: Search topic
  - Returns: Array of conversations containing the topic keyword

### UIService
Handles Discord UI elements, embeds, and pagination.

#### Methods:
- `displaySearchResults(interaction: ChatInputCommandInteraction, conversations: ConversationWithContext[], topic: string, channelName: string): Promise<void>`
  - Displays search results with interactive pagination
  - Features:
    - Conversation navigation
    - Page navigation within conversations
    - Message grouping by author
    - Auto-expiring collectors (5 minutes)
  - Parameters:
    - `interaction`: Discord interaction
    - `conversations`: Array of relevant conversations
    - `topic`: Search topic
    - `channelName`: Channel name for display

## Types

### ConversationWithContext
```typescript
interface ConversationWithContext {
    messages: Message[];     // Array of Discord messages in the conversation
    relevanceScore: number;  // AI-assigned relevance score (0-10)
}
```

### MessageGroup
```typescript
interface MessageGroup {
    author: string;      // Author's username
    content: string[];   // Array of message contents
    timestamp: number;   // Timestamp of first message
}
```

### AIAnalysisResult
```typescript
interface AIAnalysisResult {
    isRelevant: boolean;     // Whether the conversation is relevant
    relevanceScore: number;  // Relevance score (0-10)
}
```

## Constants

### EMOJIS
```typescript
const EMOJIS = {
    search: "🔍",
    conversation: "💬",
    page: "📄",
    relevance: "⭐",
    channel: "📌",
    time: "⏱️",
    error: "❌",
    success: "✅",
    warning: "⚠️"
}
```

## Example Usage

### Basic Search Command
```typescript
// Example of using the services together
const messages = await MessageService.fetchMessages(channel, limit, interaction);
const filteredMessages = MessageService.filterCommandMessages(messages, interaction);
const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
const validConversations = ConversationService.filterValidConversations(conversations);
const relevantConversations = await AIService.analyzeConversations(validConversations, topic, interaction);
await UIService.displaySearchResults(interaction, relevantConversations, topic, channel.name);
```

## Best Practices

1. **Message Fetching**
   - Always use MessageService for fetching to handle rate limits
   - Consider using smaller limits for faster responses
   - Show progress updates for large fetches

2. **Conversation Grouping**
   - Use ConversationService to maintain conversation context
   - Consider message relationships (replies, mentions)
   - Filter invalid conversations to improve quality

3. **AI Analysis**
   - Handle AI errors gracefully with fallback keyword search
   - Use batch processing for large datasets
   - Monitor and optimize batch sizes based on usage

4. **UI/UX**
   - Group messages by author for readability
   - Provide navigation controls for both conversations and pages
   - Include relevance scores and timestamps
   - Add links to original messages
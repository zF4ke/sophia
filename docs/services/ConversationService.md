[Back to Index](../API.md)

# ConversationService

The ConversationService handles message grouping, conversation analysis, and thread management in Sophia3.

## Core Features

### Message Grouping
- Time-based grouping
- Context similarity analysis
- Thread and reply handling
- Conversation boundary detection

### Thread Management
- Thread creation and tracking
- Reply chain analysis
- Context preservation
- Cross-reference handling

## Method Reference

### groupMessagesByConversation
```typescript
static groupMessagesByConversation(
  messages: Message[],
  options?: GroupingOptions
): Message[][]
```

Groups messages into conversations based on time and context.

#### Parameters:
- `messages`: Array of messages to group
- `options`: Optional configuration
  - `timeThreshold`: Maximum time gap (ms)
  - `contextSimilarity`: Required similarity score
  - `minMessages`: Minimum messages per group

#### Returns:
Array of message groups representing conversations

### createThread
```typescript
static async createThread(
  messages: Message[],
  options?: ThreadOptions
): Promise<ConversationThread>
```

Creates a thread from a group of messages.

#### Parameters:
- `messages`: Messages to include in thread
- `options`: Thread configuration
  - `timeThreshold`: Time window
  - `contextSimilarity`: Similarity threshold
  - `maxMessages`: Maximum messages

#### Returns:
Thread object with messages and metadata

### analyzeConversationBoundaries
```typescript
static analyzeConversationBoundaries(
  messages: Message[],
  options?: BoundaryOptions
): ConversationBoundary[]
```

Detects natural conversation boundaries.

#### Parameters:
- `messages`: Messages to analyze
- `options`: Analysis configuration
  - `minGap`: Minimum time gap
  - `topicShiftThreshold`: Topic change threshold
  - `participantThreshold`: Participant change weight

#### Returns:
Array of detected conversation boundaries

### mergeConversations
```typescript
static mergeConversations(
  conversations: Message[][],
  options?: MergeOptions
): Message[][]
```

Merges related conversations based on context.

#### Parameters:
- `conversations`: Conversation groups to merge
- `options`: Merge configuration
  - `similarityThreshold`: Required similarity
  - `maxTimeGap`: Maximum time between groups
  - `maxSize`: Maximum merged size

#### Returns:
Array of merged conversation groups

### extractConversationMetadata
```typescript
static extractConversationMetadata(
  conversation: Message[]
): ConversationMetadata
```

Extracts metadata from a conversation.

#### Parameters:
- `conversation`: Group of messages

#### Returns:
Metadata object with conversation details

## Integration Examples

### Basic Message Grouping
```typescript
const messages = await MessageService.fetchMessages(channel, limit);
const conversations = ConversationService.groupMessagesByConversation(messages, {
  timeThreshold: 300000, // 5 minutes
  minMessages: 2
});
```

### Thread Creation
```typescript
const thread = await ConversationService.createThread(messages, {
  timeThreshold: 600000,  // 10 minutes
  maxMessages: 50
});

await ConversationUIService.displayThread(interaction, thread);
```

### Advanced Analysis
```typescript
// Analyze conversation flow
const boundaries = ConversationService.analyzeConversationBoundaries(messages);
const groupedMessages = ConversationService.splitByBoundaries(messages, boundaries);

// Merge related conversations
const mergedGroups = ConversationService.mergeConversations(groupedMessages, {
  similarityThreshold: 0.7
});
```

## Error Handling

```typescript
try {
  const conversations = await ConversationService.groupMessagesByConversation(messages);
} catch (error) {
  if (error instanceof GroupingError) {
    // Handle grouping-specific errors
    console.error('Grouping failed:', error.message);
    return fallbackGrouping(messages);
  }
  throw error;
}
```

## Best Practices

1. **Performance**
   - Use appropriate time thresholds
   - Implement batch processing
   - Cache conversation metadata

2. **Accuracy**
   - Balance similarity thresholds
   - Consider multiple context factors
   - Validate conversation boundaries

3. **Integration**
   - Coordinate with MessageService
   - Use with AIService for analysis
   - Implement proper error handling

## Configuration

```typescript
const DEFAULT_CONFIG = {
  // Time settings (ms)
  DEFAULT_TIME_THRESHOLD: 300000,    // 5 minutes
  MAX_CONVERSATION_SPAN: 3600000,    // 1 hour
  MIN_MESSAGE_GAP: 60000,           // 1 minute
  
  // Grouping settings
  MIN_MESSAGES_PER_GROUP: 2,
  MAX_MESSAGES_PER_GROUP: 50,
  DEFAULT_SIMILARITY_THRESHOLD: 0.6,
  
  // Thread settings
  MAX_THREAD_SIZE: 100,
  THREAD_CONTEXT_WINDOW: 10
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
[Back to Index](../API.md)

# AIService

The AIService is the primary interface for AI operations using Google's Gemini API. It coordinates between specialized AI services to provide high-level operations for conversation analysis and response generation.

## Core Functionality

### Message Analysis
- Topic relevance scoring
- Conversation context analysis
- Natural language understanding
- Sentiment analysis

### Response Generation
- Context-aware responses
- Natural language generation
- Template-based formatting
- Summary generation

## Method Reference

### analyzeConversations
```typescript
static async analyzeConversations(
  conversations: Message[][],
  topic: string,
  interaction?: ChatInputCommandInteraction,
  options?: AnalysisOptions
): Promise<ConversationWithContext[]>
```

Analyzes conversations for relevance to a topic using Gemini AI.

#### Parameters:
- `conversations`: Array of message groups
- `topic`: Search topic
- `interaction`: Optional Discord interaction for progress updates
- `options`: Configuration options
  - `minRelevance`: Minimum relevance score (0-1)
  - `maxTokens`: Maximum tokens per analysis
  - `includeBots`: Whether to include bot messages

#### Returns:
Array of conversations with relevance scores and context

### generateContextualResponse
```typescript
static async generateContextualResponse(
  prompt: string,
  context: string,
  options?: ResponseOptions
): Promise<string>
```

Generates an AI response considering conversation context.

#### Parameters:
- `prompt`: User query/instruction
- `context`: Conversation context
- `options`: Generation options
  - `temperature`: Response randomness (0-1)
  - `maxTokens`: Maximum response length
  - `format`: Response format (text/markdown)

#### Returns:
Generated AI response

### fallbackKeywordSearch
```typescript
static fallbackKeywordSearch(
  conversations: Message[][],
  topic: string,
  options?: SearchOptions
): ConversationWithContext[]
```

Provides basic keyword matching when AI analysis is unavailable.

#### Parameters:
- `conversations`: Message groups
- `topic`: Search topic
- `options`: Search configuration
  - `caseSensitive`: Match case
  - `wholeWord`: Match whole words
  - `fuzzyMatch`: Allow fuzzy matching

#### Returns:
Conversations containing topic keywords

### extractKeywords
```typescript
static extractKeywords(
  text: string,
  options?: KeywordOptions
): string[]
```

Extracts meaningful keywords from text.

#### Parameters:
- `text`: Input text
- `options`: Extraction options
  - `minLength`: Minimum word length
  - `maxKeywords`: Maximum keywords
  - `languages`: Language codes for stopwords

#### Returns:
Array of extracted keywords

### optimizeContext
```typescript
static optimizeContext(
  context: string,
  maxTokens: number = 8000
): string
```

Optimizes context to fit within token limits.

#### Parameters:
- `context`: Original context
- `maxTokens`: Maximum allowed tokens

#### Returns:
Optimized context string

## Error Handling

### AIServiceError
Custom error class for AI-related failures.

```typescript
class AIServiceError extends Error {
  constructor(
    message: string,
    public readonly code: AIErrorCode,
    public readonly details?: any
  )
}
```

### Error Recovery
```typescript
try {
  return await AIService.analyzeConversations(conversations, topic);
} catch (error) {
  if (error instanceof AIServiceError) {
    switch (error.code) {
      case AIErrorCode.TokenLimit:
        return AIService.optimizeAndRetry(conversations, topic);
      case AIErrorCode.APIFailure:
        return AIService.fallbackKeywordSearch(conversations, topic);
      default:
        throw error;
    }
  }
}
```

## Integration Examples

### Basic Search
```typescript
const results = await AIService.analyzeConversations(
  conversations,
  "project planning",
  interaction
);
```

### Context-Aware Response
```typescript
const response = await AIService.generateContextualResponse(
  "What was decided?",
  channelContext,
  { temperature: 0.7 }
);
```

## Best Practices

1. **Token Management**
   - Monitor token usage
   - Implement context optimization
   - Use appropriate limits

2. **Error Handling**
   - Implement fallbacks
   - Handle rate limits
   - Log errors appropriately

3. **Performance**
   - Cache results when possible
   - Use batch processing
   - Optimize context size

## Configuration

### Default Settings
```typescript
const DEFAULT_CONFIG = {
  maxTokens: 8000,
  temperature: 0.4,
  minRelevance: 0.6,
  cacheTimeout: 3600,
  retryAttempts: 3
};
```

### Rate Limits
- Requests per minute: 60
- Tokens per request: 8000
- Daily token limit: 1M

For implementation examples, see the [Examples Guide](../guides/Examples.md).
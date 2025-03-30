[Back to Index](../API.md)

# ContextManagementService

The ContextManagementService handles the preparation, optimization, and management of conversation context for AI operations.

## Core Features

### Context Management
- Context window optimization
- Token limit management
- Relevance scoring
- Context preservation

### Memory Management
- Short-term context cache
- Long-term memory storage
- Context prioritization
- Memory cleanup

## Method Reference

### getChannelContext
```typescript
static async getChannelContext(
  channel: TextChannel,
  options?: ContextOptions
): Promise<string>
```

Retrieves and optimizes context from a channel.

#### Parameters:
- `channel`: Discord text channel
- `options`: Context settings
  - `messageLimit`: Max messages to fetch
  - `maxTokens`: Token limit
  - `relevanceThreshold`: Min relevance score

#### Returns:
Optimized context string

### optimizeContext
```typescript
static optimizeContext(
  context: string,
  options: OptimizationOptions
): string
```

Optimizes context for token limits.

#### Parameters:
- `context`: Raw context string
- `options`: Optimization settings
  - `maxTokens`: Maximum tokens
  - `preserveRecent`: Prioritize recent
  - `strategy`: Optimization strategy

#### Returns:
Optimized context string

### createMemoryCache
```typescript
static createMemoryCache(
  key: string,
  options?: CacheOptions
): ContextCache
```

Creates a context memory cache.

#### Parameters:
- `key`: Cache identifier
- `options`: Cache configuration
  - `maxSize`: Maximum entries
  - `ttl`: Time to live
  - `priority`: Cache priority

#### Returns:
Context cache instance

### scoreRelevance
```typescript
static scoreRelevance(
  content: string,
  topic: string,
  options?: ScoringOptions
): number
```

Scores content relevance to a topic.

#### Parameters:
- `content`: Content to score
- `topic`: Reference topic
- `options`: Scoring options
  - `algorithm`: Scoring method
  - `threshold`: Minimum score
  - `weights`: Feature weights

#### Returns:
Relevance score (0-1)

## Integration Examples

### Basic Context Management
```typescript
// Get channel context
const context = await ContextManagementService.getChannelContext(channel, {
  messageLimit: 1000,
  maxTokens: 6000,
  relevanceThreshold: 0.6
});

// Optimize for AI
const optimized = ContextManagementService.optimizeContext(context, {
  maxTokens: 4000,
  preserveRecent: true
});
```

### Memory Management
```typescript
// Create context cache
const cache = ContextManagementService.createMemoryCache('channel-123', {
  maxSize: 1000,
  ttl: '1h'
});

// Store context
await cache.set('conversation-1', {
  content: context,
  timestamp: Date.now(),
  score: 0.8
});
```

### Relevance Scoring
```typescript
// Score content relevance
const score = ContextManagementService.scoreRelevance(
  messageContent,
  searchTopic,
  {
    algorithm: 'cosine',
    threshold: 0.5
  }
);

if (score > 0.7) {
  await includeInContext(messageContent);
}
```

## Error Handling

### Context Errors
```typescript
try {
  const context = await ContextManagementService.getChannelContext(channel);
} catch (error) {
  if (error instanceof ContextLimitError) {
    return await getReducedContext(channel);
  }
  throw error;
}
```

### Memory Management
```typescript
try {
  await cache.set(key, value);
} catch (error) {
  if (error instanceof CacheFullError) {
    await cache.cleanup();
    await cache.set(key, value);
  }
}
```

## Best Practices

1. **Context Optimization**
   - Balance context size
   - Preserve important content
   - Implement smart truncation

2. **Memory Management**
   - Regular cache cleanup
   - Priority-based retention
   - Efficient storage use

3. **Performance**
   - Optimize token usage
   - Cache frequent contexts
   - Batch operations

## Configuration

```typescript
const CONTEXT_CONFIG = {
  // Context limits
  MAX_TOKENS: 8000,
  MAX_MESSAGES: 2000,
  MIN_RELEVANCE: 0.6,
  
  // Memory settings
  CACHE_SIZE: 10000,
  CACHE_TTL: '1h',
  CLEANUP_INTERVAL: '5m',
  
  // Optimization
  PRESERVATION_RATIO: 0.7,
  RECENCY_WEIGHT: 0.3,
  RELEVANCE_WEIGHT: 0.7,
  
  // Algorithms
  SCORING_METHOD: 'cosine',
  OPTIMIZATION_STRATEGY: 'balanced'
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
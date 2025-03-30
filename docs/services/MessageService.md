[Back to Index](../API.md)

# MessageService

The MessageService handles message operations including fetching, caching, and filtering Discord messages.

## Core Features

### Message Operations
- Message fetching with pagination
- Efficient caching system
- Message filtering
- Batch processing

### Cache Management
- Intelligent cache storage
- Automatic cache cleanup
- Cache invalidation
- Memory optimization

## Method Reference

### fetchMessages
```typescript
static async fetchMessages(
  channel: TextChannel,
  options: FetchOptions
): Promise<Message[]>
```

Fetches messages from a Discord channel.

#### Parameters:
- `channel`: Discord text channel
- `options`: Fetch configuration
  - `limit`: Maximum messages
  - `before`: Message ID to fetch before
  - `after`: Message ID to fetch after
  - `around`: Message ID to fetch around

#### Returns:
Array of fetched messages

### cacheMessages
```typescript
static async cacheMessages(
  messages: Message[],
  options?: CacheOptions
): Promise<void>
```

Stores messages in the cache.

#### Parameters:
- `messages`: Messages to cache
- `options`: Cache settings
  - `expiry`: Cache lifetime
  - `priority`: Cache priority
  - `tags`: Cache tags

### getCachedMessages
```typescript
static async getCachedMessages(
  channelId: string,
  options?: CacheQueryOptions
): Promise<Message[]>
```

Retrieves messages from cache.

#### Parameters:
- `channelId`: Channel ID
- `options`: Query options
  - `limit`: Maximum results
  - `tags`: Filter by tags
  - `includeExpired`: Include expired entries

#### Returns:
Array of cached messages

### filterMessages
```typescript
static filterMessages(
  messages: Message[],
  filters: MessageFilter
): Message[]
```

Filters messages based on criteria.

#### Parameters:
- `messages`: Messages to filter
- `filters`: Filter criteria
  - `content`: Content filter
  - `authors`: Author filter
  - `timestamp`: Time range
  - `hasAttachments`: Attachment filter

#### Returns:
Filtered message array

## Integration Examples

### Basic Message Fetch
```typescript
const messages = await MessageService.fetchMessages(channel, {
  limit: 1000,
  before: lastMessageId
});

const filtered = MessageService.filterMessages(messages, {
  authors: ['user1', 'user2'],
  hasAttachments: true
});
```

### Cache Usage
```typescript
// Cache fetched messages
await MessageService.cacheMessages(messages, {
  expiry: '1h',
  tags: ['search', channelId]
});

// Retrieve from cache
const cached = await MessageService.getCachedMessages(channelId, {
  limit: 100,
  tags: ['search']
});
```

### Batch Processing
```typescript
const batches = MessageService.createBatches(messages, {
  size: 100,
  overlap: 10
});

for (const batch of batches) {
  await processMessageBatch(batch);
}
```

## Error Handling

### Fetch Errors
```typescript
try {
  const messages = await MessageService.fetchMessages(channel, options);
} catch (error) {
  if (error instanceof DiscordAPIError) {
    console.error('API error:', error.message);
    return await MessageService.getCachedMessages(channel.id);
  }
  throw error;
}
```

### Cache Management
```typescript
try {
  await MessageService.cacheMessages(messages);
} catch (error) {
  if (error instanceof CacheFullError) {
    await MessageService.cleanupCache();
    await MessageService.cacheMessages(messages);
  }
}
```

## Best Practices

1. **Performance**
   - Use appropriate fetch limits
   - Implement efficient caching
   - Process in batches

2. **Cache Management**
   - Set reasonable expiry times
   - Regular cache cleanup
   - Priority-based eviction

3. **Error Recovery**
   - Handle API limits
   - Implement retries
   - Use cache fallback

## Configuration

```typescript
const MESSAGE_CONFIG = {
  // Fetch settings
  DEFAULT_FETCH_LIMIT: 2000,
  MAX_FETCH_SIZE: 100,
  FETCH_DELAY: 1000,
  
  // Cache settings
  CACHE_SIZE: 10000,
  DEFAULT_EXPIRY: '1h',
  CLEANUP_INTERVAL: '5m',
  
  // Batch settings
  BATCH_SIZE: 100,
  DEFAULT_OVERLAP: 10,
  MAX_BATCHES: 20
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
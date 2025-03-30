[Back to Index](../API.md)

# MessageService

The MessageService handles Discord message fetching and filtering operations. It provides robust methods for retrieving messages from Discord channels with built-in rate limiting, progress reporting, filtering capabilities, and an advanced persistent caching system.

## Properties

### Cache-Related Constants

```typescript
private static readonly USE_CACHE = true;              // Enable/disable caching
private static readonly CACHE_EXPIRY = 30 * 24 * 60 * 60 * 1000;  // 30 days cache expiry
private static readonly SERVICE_NAME = 'message';       // Service name for file organization
private static readonly messageCache = new Map<string, MessageCache>();
```

These properties control the message caching behavior:
- `USE_CACHE`: Toggle to enable/disable the caching system
- `CACHE_EXPIRY`: Time in milliseconds before cached messages expire (set to 30 days)
- `SERVICE_NAME`: Used for file path organization with FileSystemService
- `messageCache`: In-memory storage for cached messages by channel ID

## Methods

### `initializeCache`

```typescript
public static initializeCache(): void
```

Initializes the message cache from disk storage on bot startup.

#### Features:
- Automatically called when the bot starts
- Loads previously cached messages from disk
- Removes expired cache files
- Uses FileSystemService for cache file operations

#### Implementation Details:
- Checks each cached file for validity before loading
- Reports the number of channels with successfully loaded caches
- Handles and reports errors during cache initialization

### `fetchMessages`

```typescript
static async fetchMessages(
  channel: TextChannel, 
  limit: number, 
  interaction: ChatInputCommandInteraction
): Promise<Message[]>
```

Fetches messages from a Discord channel with progress updates.

#### Features:
- Advanced intelligent caching system with disk persistence
- Smart fetching algorithm that prioritizes getting only new messages
- Persistent cache checking that looks for cached messages in every batch
- Seamlessly merges new messages with cached messages at any point in history
- Supports fetching up to 20,000 messages
- Handles Discord API rate limiting automatically
- Shows progress updates during fetching via interaction responses
- Intelligently detects when channel history end is reached
- Includes retry logic for transient errors
- Uses UIService.formatStatusMessage for consistent status updates

#### Parameters:
- `channel: TextChannel` - The Discord text channel to fetch messages from
- `limit: number` - Maximum number of messages to fetch
- `interaction: ChatInputCommandInteraction` - Discord interaction for progress updates

#### Returns:
- `Promise<Message[]>` - Array of Discord messages

#### Example:
```typescript
const messages = await MessageService.fetchMessages(channel, 1000, interaction);
console.log(`Fetched ${messages.length} messages`);
```

#### Implementation Details:
- First attempts to fetch newest messages and compare with cache
- When overlap is found, combines new messages with cached messages
- Continues checking for cached messages in subsequent batches
- Never gives up looking for matching messages in cache, even after many batches
- Reconstructs Message-like objects from cached data
- Only fetches additional messages when necessary
- Automatically updates cache with new messages
- Saves cache to disk for persistence
- Uses batch processing with `MAX_BATCH_SIZE` (100) messages per request
- Provides descriptive progress updates using UIService.formatStatusMessage
- Uses backtick-formatted status messages for in-progress updates
- Plain text formatting for final status message
- Handles permission errors (code 50001) by throwing clear errors
- Implements rate limit handling by respecting Discord's retry-after response

### `filterCommandMessages`

```typescript
static filterCommandMessages(
  messages: Message[], 
  interaction: ChatInputCommandInteraction
): Message[]
```

Filters out bot commands and recent messages from the command author.

#### Features:
- Removes bot own messages to reduce noise
- Filters out command invocations from the current interaction user
- Maintains context by keeping older messages from the command author

#### Parameters:
- `messages: Message[]` - Array of messages to filter
- `interaction: ChatInputCommandInteraction` - Current command interaction

#### Returns:
- `Message[]` - Filtered array of messages

#### Example:
```typescript
const messages = await MessageService.fetchMessages(channel, 1000, interaction);
const filteredMessages = MessageService.filterCommandMessages(messages, interaction);
console.log(`Filtered out ${messages.length - filteredMessages.length} messages`);
```

#### Implementation Details:
- Removes messages from the bot itself
- Filters out messages from the command author within the last 10 seconds
- Preserves older messages from all users, including command author

### `clearCache`

```typescript
static clearCache(channelId?: string): void
```

Clears the message cache for a specific channel or all channels, both from memory and disk.

#### Features:
- Selectively clear cache for a specific channel
- Option to clear the entire cache
- Removes cache files from disk to free up storage
- Useful when channel content has changed significantly

#### Parameters:
- `channelId?: string` - Optional channel ID to clear cache for. If not provided, clears all cache.

#### Example:
```typescript
// Clear cache for a specific channel
MessageService.clearCache('123456789012345678');

// Clear all cached messages
MessageService.clearCache();
```

## Internal Methods

### `saveCache`

```typescript
private static saveCache(channelId: string): void
```

Saves the in-memory cache to disk for persistence.

### `messageToCache`

```typescript
private static messageToCache(message: Message): MessageCacheItem
```

Converts a Discord Message object to a simplified format for storage.

### `updateCache`

```typescript
private static updateCache(channelId: string, messages: Message[]): void
```

Updates the message cache with new messages and saves to disk.

### `reconstructMessages`

```typescript
private static reconstructMessages(channel: TextChannel, cachedMessages: MessageCacheItem[]): Message[]
```

Reconstructs Message-like objects from cached data.
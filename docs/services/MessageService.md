[Back to Index](../API.md)

# MessageService

The MessageService handles Discord message fetching and filtering operations. It provides robust methods for retrieving messages from Discord channels with built-in rate limiting, progress reporting, and filtering capabilities.

## Methods

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
- Supports fetching up to 20,000 messages
- Handles Discord API rate limiting automatically
- Shows progress updates during fetching via interaction responses
- Intelligently detects when channel history end is reached
- Includes retry logic for transient errors

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
- Uses batch processing with `MAX_BATCH_SIZE` (100) messages per request
- Provides progress updates at `UPDATE_INTERVAL` (5000ms) intervals
- Detects channel end when batch size falls below `MIN_MESSAGES_FOR_CHANNEL_END` (25)
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
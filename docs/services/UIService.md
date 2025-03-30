[Back to Index](../API.md)

# UIService

The UIService handles the creation and management of Discord UI elements, such as embeds and interactive buttons for pagination. It specializes in displaying conversation search results with rich formatting and navigation capabilities.

## Properties

### Configuration Constants

```typescript
private static readonly MAX_MESSAGES_PER_PAGE = 5;
private static readonly COLLECTOR_TIMEOUT = 300000; // 5 minutes
private static readonly MAX_FIELD_VALUE_LENGTH = 1024;
private static readonly MAX_FIELD_NAME_LENGTH = 256;
```

These constants define various UI and pagination settings:
- Maximum messages displayed per page
- Component collector timeout (5 minutes)
- Discord embed field size limits

## Methods

### `formatStatusMessage`

```typescript
static formatStatusMessage(emoji: string, message: string, useBackticks = true): string
```

Creates a formatted status message with an emoji and optional backtick formatting.

#### Features:
- Consistent formatting for status messages throughout the application
- Optional backtick formatting for in-progress updates
- Uses emoji for visual status indicators

#### Parameters:
- `emoji: string` - The emoji character to use
- `message: string` - The status message text
- `useBackticks: boolean` - Whether to format with backticks (default: true)

#### Returns:
- `string` - Formatted status message

#### Example:
```typescript
// With backticks (for progress updates)
const loadingMessage = UIService.formatStatusMessage(EMOJIS.loading, "Carregando mensagens...");
// Without backticks (for final results)
const completeMessage = UIService.formatStatusMessage(EMOJIS.complete, "Busca completa!", false);

await interaction.editReply(loadingMessage);
```

### `displaySearchResults`

```typescript
static async displaySearchResults(
  interaction: ChatInputCommandInteraction,
  conversations: ConversationWithContext[],
  topic: string,
  channelName: string,
  ephemeral: boolean = false
): Promise<void>
```

Displays conversation search results with interactive pagination controls.

#### Features:
- Conversation navigation (previous/next conversation)
- Page navigation within conversations (previous/next page)
- Message grouping by author for readability
- Auto-expiring UI components (5 minutes)
- Supports ephemeral (private) responses

#### Parameters:
- `interaction: ChatInputCommandInteraction` - Discord interaction to respond to
- `conversations: ConversationWithContext[]` - Array of relevant conversations with context
- `topic: string` - Search topic for display
- `channelName: string` - Channel name for display
- `ephemeral: boolean` - Whether to show the results only to the command user (default: false)

#### Example:
```typescript
const relevantConversations = await AIService.analyzeConversations(validConversations, topic, interaction);
await UIService.displaySearchResults(interaction, relevantConversations, topic, channel.name);
```

### `createEmbed` (Private)

```typescript
private static createEmbed(
  conversation: ConversationWithContext,
  startIndex: number,
  topic: string,
  channelName: string,
  totalConversations: number,
  currentConvIndex: number
): EmbedBuilder
```

Creates a rich embed for displaying conversation content.

#### Features:
- Displays conversation metadata (relevance score, channel, links)
- Groups messages by author
- Handles large messages by splitting into chunks
- Truncates long content to meet Discord's limits
- Shows pagination information

#### Parameters:
- `conversation: ConversationWithContext` - The conversation to display
- `startIndex: number` - Starting message index for pagination
- `topic: string` - Search topic
- `channelName: string` - Channel name
- `totalConversations: number` - Total number of conversations
- `currentConvIndex: number` - Current conversation index

#### Returns:
- `EmbedBuilder` - Discord embed for displaying the conversation

### `splitContentIntoChunks` (Private)

```typescript
private static splitContentIntoChunks(content: string): string[]
```

Splits large message content into smaller chunks to fit within Discord embed field limits.

#### Features:
- Smart splitting at word boundaries when possible
- Ensures chunks are within Discord's field value size limits
- Preserves message formatting where possible

#### Parameters:
- `content: string` - Message content to split

#### Returns:
- `string[]` - Array of content chunks

### `createButtonRow` (Private)

```typescript
private static createButtonRow(
  currentConvIndex: number,
  currentMsgIndex: number,
  conversations: ConversationWithContext[]
): ActionRowBuilder<ButtonBuilder>
```

Creates navigation buttons for conversation and page navigation.

#### Features:
- Previous/next conversation buttons
- Previous/next page buttons within the current conversation
- Dynamic button disabling based on navigation state

#### Parameters:
- `currentConvIndex: number` - Current conversation index
- `currentMsgIndex: number` - Current message index for pagination
- `conversations: ConversationWithContext[]` - Array of conversations

#### Returns:
- `ActionRowBuilder<ButtonBuilder>` - Row of navigation buttons

### `groupMessagesByAuthor` (Private)

```typescript
private static groupMessagesByAuthor(messages: Message[]): MessageGroup[]
```

Groups messages by author to improve readability.

#### Features:
- Combines consecutive messages from the same author
- Preserves chronological order
- Includes author name and timestamp information

#### Parameters:
- `messages: Message[]` - Messages to group

#### Returns:
- `MessageGroup[]` - Array of message groups by author

### `handleCollector` (Private)

```typescript
private static handleCollector(
  collector: any,
  interaction: ChatInputCommandInteraction,
  conversations: ConversationWithContext[],
  topic: string,
  channelName: string,
  state: { currentConvIndex: number; currentMsgIndex: number },
  ephemeral: boolean = false
): void
```

Sets up a component collector to handle button interactions.

#### Features:
- Handles navigation button clicks
- Updates UI based on user interaction
- Enforces permissions (only command author and admins can interact)
- Handles collector timeout by disabling buttons
- Updates the displayed message when navigation occurs

#### Parameters:
- `collector: any` - Discord component collector
- `interaction: ChatInputCommandInteraction` - Original command interaction
- `conversations: ConversationWithContext[]` - Conversations to display
- `topic: string` - Search topic
- `channelName: string` - Channel name
- `state: { currentConvIndex: number; currentMsgIndex: number }` - Navigation state
- `ephemeral: boolean` - Whether the response is ephemeral

## Best Practices

- Use `formatStatusMessage` for all status updates to maintain consistent styling
- Use backtick formatting for in-progress updates and remove it for final results
- Use this service as the final step in search and display workflows
- Consider using ephemeral responses in busy channels to reduce clutter
- Remember that collectors automatically expire after the timeout period
- Use the provided navigation system rather than creating multiple results messages

---

# UI Service Documentation

The UI Service system is composed of a base `UIService` class and specialized UI services that extend it. This architecture provides reusable UI components and functionality while allowing for specialized implementations.

## Base UIService

The base `UIService` class provides core functionality for creating interactive Discord message components and handling pagination.

### Public Methods

#### `formatStatusMessage(emoji: string, message: string, useBackticks = true)`
Creates a formatted status message with consistent styling for use throughout the application.

### Protected Methods

#### `createNavigationRow(buttons: NavigationButton[])`
Creates a row of buttons for navigation in paginated displays.

#### `addMessageGroupFields(embed: EmbedBuilder, messages: Message[])`
Adds grouped message fields to an embed, handling message chunking and formatting.

#### `splitContentIntoChunks(content: string)`
Splits content into chunks that fit within Discord's field value limits.

#### `groupMessagesByAuthor(messages: Message[])`
Groups messages by their authors for cleaner display.

#### `setupInteractionCollector<T>()`
Sets up an interaction collector for handling button clicks in paginated displays.

### Constants

- `MAX_ITEMS_PER_PAGE`: 5
- `COLLECTOR_TIMEOUT`: 300000 (5 minutes)
- `MAX_FIELD_VALUE_LENGTH`: 1024
- `MAX_FIELD_NAME_LENGTH`: 256
- `DEFAULT_COLOR`: 0x7289da

### Interfaces

#### NavigationButton
```typescript
{
    customId: string;
    label: string;
    style: 'Primary' | 'Secondary' | 'Success' | 'Danger';
    disabled: boolean;
}
```

#### NavigationState
```typescript
{
    currentConvIndex: number;
    currentMsgIndex: number;
}
```

## Error Handling

The service includes robust error handling for:
- Expired messages
- Unknown message errors
- Interaction timeouts
- Permission validation

## Best Practices

1. Always extend UIService for specialized UI implementations
2. Use the provided interfaces for type safety
3. Implement error handling using the base service's mechanisms
4. Follow the pagination pattern for large datasets
5. Use formatStatusMessage for consistent status update styling
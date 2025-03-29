# ConversationUIService Documentation

The `ConversationUIService` extends the base `UIService` to provide specialized UI functionality for displaying Discord conversations with rich formatting and interactive navigation.

## Overview

ConversationUIService specializes in displaying conversations with pagination controls, metadata, and interactive navigation. It inherits core UI functionality from UIService while adding conversation-specific features.

## Main Features

### `displayConversations`

```typescript
public static async displayConversations(
    interaction: ChatInputCommandInteraction,
    conversations: ConversationWithContext[],
    topic: string,
    channelName: string,
    ephemeral: boolean = false
): Promise<void>
```

Displays paginated conversations with the following features:
- Conversation-level navigation
- Page-level navigation within conversations
- Metadata display (scores, channel information)
- Direct links to original messages
- Channel context information
- Support for ephemeral (private) responses

#### Parameters:
- `interaction` - The Discord interaction to respond to
- `conversations` - Array of conversations with context
- `topic` - The context or topic being displayed
- `channelName` - The name of the channel
- `ephemeral` - Whether to show results only to the command user

## Implementation Details

### Embed Creation
- Dynamic title with context/topic
- Metadata display
- Channel information
- Jump links to original messages
- Paginated message display
- Timestamp information

### Navigation
- Primary buttons for conversation navigation
- Secondary buttons for page navigation
- Dynamic button states based on current position
- Proper handling of navigation limits

## Usage Example

```typescript
// Display conversations with pagination
const conversations = await getConversations();
await ConversationUIService.displayConversations(
    interaction,
    conversations,
    "Topic or Context",
    "channel-name",
    true // for ephemeral response
);
```

## Error Handling

The service includes specific handling for:
- Invalid interactions
- Expired messages
- Navigation boundaries
- Permission errors

## Best Practices

1. Use ephemeral responses when appropriate to reduce channel clutter
2. Ensure conversations are properly formatted before display
3. Consider any relevant metadata for the conversations
4. Handle navigation timeouts gracefully
5. Provide clear context in the embed description

## Related Services
- Base UIService for core functionality
- ConversationService for message grouping
- MessageService for message handling
[Back to Index](../API.md)

# Sophia3 Examples

## Displaying Conversations

### Basic Conversation Display

```typescript
import { ConversationUIService } from "../services/ui/ConversationUIService";

// Display any group of conversations with context
await ConversationUIService.displayConversations(
    interaction,
    conversations,
    "Context or Topic",
    channelName
);
```

### Search Results Display

```typescript
// Using ConversationUIService for search results
const searchResults = await AIService.analyzeConversations(conversations, topic, interaction);
await ConversationUIService.displayConversations(
    interaction, 
    searchResults, 
    topic,
    channel.name,
    true // ephemeral
);
```

## Working with Message Groups

### Grouping Messages

```typescript
const messages = await MessageService.fetchMessages(channel, limit);
const conversations = ConversationService.groupMessagesByConversation(messages);
```

## Best Practices

### Ephemeral Messages
Always consider using ephemeral messages in busy channels:
```typescript
await ConversationUIService.displayConversations(
    interaction,
    results,
    topic,
    channelName,
    true // ephemeral
);
```

### Error Handling
The UI services include built-in error handling for common scenarios:
- Expired interactions
- Unknown messages
- Permission issues
- Navigation limits

### Navigation Design
The UI system provides a consistent navigation pattern:
- Primary buttons for major navigation (between conversations)
- Secondary buttons for minor navigation (between pages)
- Clear visual feedback on navigation limits

## Advanced Usage

### Custom Message Grouping
```typescript
export class CustomUIService extends UIService {
    protected static customGroupMessages(messages: Message[]): CustomGroup[] {
        // Your custom grouping logic
        return customGroups;
    }
}
```

### Extended Navigation
```typescript
const customButtons: NavigationButton[] = [
    {
        customId: 'custom_action',
        label: 'Custom Action',
        style: 'Primary',
        disabled: false
    },
    // ... other buttons
];
```
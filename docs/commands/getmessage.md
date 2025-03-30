# GetMessage Command

The getmessage command retrieves specific messages and their surrounding context from a channel.

## Usage

```
/getmessage [channel] [number] [ephemeral]
```

## Parameters

### Required
- `channel`: The channel containing the message
- `number`: The message number (position from channel start)

### Optional
- `ephemeral`: Show result only to you (default: true)

## Examples

### Basic Usage
```
/getmessage channel:#announcements number:50
```

### Private Retrieval
```
/getmessage channel:#team-updates number:100 ephemeral:true
```

## Permissions

- Requires READ_MESSAGES in target channel
- Rate limited to 30 uses per hour per user
- No special permissions for ephemeral responses

## Response Format

### Success
```typescript
{
  embeds: [
    {
      title: "Message #{number}",
      description: "From {channel}",
      fields: [
        {
          name: "Content",
          value: "{message_content}"
        },
        {
          name: "Context",
          value: "Previous and following messages"
        }
      ],
      footer: {
        text: "Sent by {author} at {timestamp}"
      }
    }
  ]
}
```

### Error States
- Message not found
- Invalid number
- Channel not found
- Permission denied

## Integration

### With Search Command
```
// Find relevant messages
/search topic:"announcement" channel:#general
// Get specific message
/getmessage channel:#general number:150
```

### With Context Command
```
// Get specific message
/getmessage channel:#project number:75
// Get context around it
/context prompt:"Explain this decision" channel:#project
```

## Best Practices

1. **Message Selection**
   - Use search first if unsure
   - Consider context needs
   - Verify message exists

2. **Privacy**
   - Use ephemeral for sensitive info
   - Check channel permissions
   - Consider audience

3. **Performance**
   - Cache frequently accessed channels
   - Use reasonable numbers
   - Handle pagination

## Configuration

```typescript
const GETMESSAGE_CONFIG = {
  // Message settings
  CONTEXT_MESSAGES: 5,
  MAX_MESSAGE_LENGTH: 2000,
  
  // Rate limiting
  MAX_USES_PER_HOUR: 30,
  COOLDOWN: '10s',
  
  // Display
  SHOW_TIMESTAMPS: true,
  SHOW_REACTIONS: true,
  INCLUDE_EMBEDS: true
};
```

For implementation details, see the [API Documentation](../API.md).
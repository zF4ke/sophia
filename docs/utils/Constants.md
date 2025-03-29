[Back to Index](../API.md)

# Constants

This document describes the important constants used throughout the Sophia3 application.

## Emoji Constants

The `EMOJIS` constant provides a consistent set of emojis used across the application UI for various indicators and actions.

```typescript
export const EMOJIS = {
    search: "🔍",
    conversation: "💬",
    page: "📄",
    relevance: "⭐",
    channel: "📌",
    time: "⏱️",
    error: "❌",
    success: "✅",
    warning: "⚠️"
} as const;
```

These emojis are used consistently across the application to indicate:

- **search**: Search operations and results
- **conversation**: Conversation groups
- **page**: Pagination and page navigation
- **relevance**: Relevance scores in search results
- **channel**: Channel references
- **time**: Timestamps and timing information
- **error**: Error messages
- **success**: Success messages and confirmations
- **warning**: Warning messages and alerts

## Admin ID Constants

The `ADMIN_IDS` array contains the Discord user IDs of administrators who have elevated permissions.

```typescript
export const ADMIN_IDS = [
    "676156690395037713",
    "111591984245780480"
];
```

These IDs are used for authorization checks in commands like `search` and `context` that are restricted to admin users.

## Response Messages

### Busy Messages

The `busy` array contains randomized response messages used when the bot is unable to process a request.

```typescript
export const busy = [
    "Eae, eu to ocupada agora. Logo mais eu volto.",
    "Oi, eu to ocupada agora. Logo mais eu volto.",
    "Beleza. Olha, agora não da pra eu te ajudar. Logo mais eu volto.",
    "Fala, olha eu to ocupada agora, mas mais tarde eu volto.",
    "Oi! No momento to ocupada, mas depois a gente se fala.",
    // Additional messages...
];
```

These messages are used primarily in the `oi` command to provide friendly responses when the bot is not available.

## Usage Examples

### Using Emojis in Messages

```typescript
// Success message with emoji
await interaction.reply(`${EMOJIS.success} Operation completed successfully!`);

// Error message with emoji
await interaction.reply(`${EMOJIS.error} An error occurred: ${errorMessage}`);

// Progress update with emoji
await interaction.editReply(`${EMOJIS.search} Searching... (${progress}%)`);
```

### Admin Permission Check

```typescript
// Check if user has admin permissions
if (!ADMIN_IDS.includes(interaction.user.id)) {
    return await interaction.reply({
        content: `${EMOJIS.error} This command is available only to administrators.`,
        flags: MessageFlags.Ephemeral
    });
}
```

### Random Busy Message

```typescript
// Get a random busy message
const randomIndex = Math.floor(Math.random() * busy.length);
const randomMessage = busy[randomIndex];

// Send the message
await interaction.reply({
    content: randomMessage,
});
```
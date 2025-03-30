[Back to Index](../../API.md)

# ConversationUIService

The ConversationUIService provides specialized UI components for displaying conversations and search results.

## Core Features

### Display Components
- Conversation embeds
- Search result views
- Thread displays
- Message formatting

### Interaction Handling
- Result pagination
- Navigation controls
- Action buttons
- Progress indicators

## Method Reference

### displayConversations
```typescript
static async displayConversations(
  interaction: CommandInteraction,
  conversations: ConversationWithContext[],
  options?: DisplayOptions
): Promise<void>
```

Displays conversations with interactive controls.

#### Parameters:
- `interaction`: Discord interaction
- `conversations`: Conversations to display
- `options`: Display settings
  - `itemsPerPage`: Results per page
  - `showMetadata`: Include metadata
  - `format`: Display format
  - `ephemeral`: Private response

### createConversationEmbed
```typescript
static createConversationEmbed(
  conversation: ConversationWithContext,
  options?: EmbedOptions
): EmbedBuilder
```

Creates an embed for a conversation.

#### Parameters:
- `conversation`: Conversation data
- `options`: Embed settings
  - `color`: Embed color
  - `showTimestamps`: Include times
  - `includeContext`: Show context
  - `maxLength`: Content limit

#### Returns:
Configured EmbedBuilder

### handlePagination
```typescript
static async handlePagination(
  interaction: ButtonInteraction,
  conversations: ConversationWithContext[],
  options?: PaginationOptions
): Promise<void>
```

Handles pagination interactions.

#### Parameters:
- `interaction`: Button interaction
- `conversations`: Full result set
- `options`: Pagination settings
  - `currentPage`: Active page
  - `itemsPerPage`: Items per page
  - `timeout`: Control timeout

### updateDisplay
```typescript
static async updateDisplay(
  interaction: CommandInteraction,
  content: DisplayContent,
  options?: UpdateOptions
): Promise<void>
```

Updates conversation display.

#### Parameters:
- `interaction`: Command interaction
- `content`: New display content
- `options`: Update settings
  - `edit`: Edit existing
  - `components`: UI components
  - `ephemeral`: Private update

## Integration Examples

### Basic Display
```typescript
// Display search results
await ConversationUIService.displayConversations(
  interaction,
  searchResults,
  {
    itemsPerPage: 5,
    showMetadata: true,
    ephemeral: true
  }
);
```

### Custom Embed
```typescript
// Create conversation embed
const embed = ConversationUIService.createConversationEmbed(
  conversation,
  {
    color: UI_CONSTANTS.COLORS.PRIMARY,
    showTimestamps: true,
    maxLength: 1024
  }
);

await interaction.reply({ embeds: [embed] });
```

### Interactive Navigation
```typescript
// Handle navigation
await ConversationUIService.handlePagination(
  buttonInteraction,
  allResults,
  {
    currentPage: 0,
    itemsPerPage: 5,
    timeout: 300000
  }
);
```

## Error Handling

### Display Errors
```typescript
try {
  await ConversationUIService.displayConversations(interaction, results);
} catch (error) {
  if (error instanceof DisplayError) {
    await interaction.reply({
      content: 'Could not display results: ' + error.message,
      ephemeral: true
    });
    return;
  }
  throw error;
}
```

### Interaction Timeouts
```typescript
try {
  await ConversationUIService.handlePagination(interaction, results);
} catch (error) {
  if (error instanceof InteractionTimeoutError) {
    await interaction.editReply({
      components: [] // Remove buttons
    });
  }
}
```

## Best Practices

1. **User Experience**
   - Clear navigation
   - Consistent formatting
   - Informative metadata

2. **Performance**
   - Optimize embed content
   - Efficient pagination
   - Handle large datasets

3. **Interaction Design**
   - Intuitive controls
   - Proper timeouts
   - Error feedback

## Configuration

```typescript
const UI_CONFIG = {
  // Display settings
  ITEMS_PER_PAGE: 5,
  MAX_PAGES: 20,
  BUTTON_TIMEOUT: 300000,
  
  // Content limits
  MAX_EMBED_LENGTH: 4096,
  MAX_FIELD_LENGTH: 1024,
  MAX_TITLE_LENGTH: 256,
  
  // Style settings
  COLORS: {
    DEFAULT: 0x0099ff,
    HIGHLIGHT: 0x00ff00,
    ERROR: 0xff0000
  },
  
  // Button settings
  NAVIGATION_STYLE: 'PRIMARY',
  ACTION_STYLE: 'SECONDARY',
  EMOJI_ENABLED: true
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
[Back to Index](../API.md)

# UIService

The UIService provides base UI functionality and components for Discord message display and interaction handling.

## Core Features

### Message Display
- Rich embed generation
- Pagination controls
- Interactive buttons
- Progress indicators

### Interaction Handling
- Button interactions
- Command responses
- Error displays
- Loading states

## Method Reference

### createEmbed
```typescript
static createEmbed(
  options: EmbedOptions
): EmbedBuilder
```

Creates a Discord embed with standardized styling.

#### Parameters:
- `options`: Embed configuration
  - `title`: Embed title
  - `description`: Main content
  - `color`: Accent color
  - `footer`: Footer text
  - `timestamp`: Show timestamp

#### Returns:
Configured EmbedBuilder instance

### createPaginatedView
```typescript
static async createPaginatedView<T>(
  items: T[],
  options: PaginationOptions,
  renderer: (items: T[]) => EmbedBuilder
): Promise<InteractionResponse>
```

Creates a paginated view of items.

#### Parameters:
- `items`: Array of items to display
- `options`: Pagination settings
  - `itemsPerPage`: Items per page
  - `timeout`: Button timeout
  - `initialPage`: Starting page
- `renderer`: Function to render items

#### Returns:
Interaction response with pagination

### createButtonRow
```typescript
static createButtonRow(
  buttons: NavigationButton[]
): ActionRowBuilder<ButtonBuilder>
```

Creates a row of interaction buttons.

#### Parameters:
- `buttons`: Button configurations
  - `customId`: Button identifier
  - `label`: Button text
  - `style`: Button style
  - `disabled`: Disabled state
  - `emoji`: Optional emoji

#### Returns:
ActionRow with configured buttons

### showProgress
```typescript
static async showProgress(
  interaction: CommandInteraction,
  message: string,
  progress: number
): Promise<void>
```

Updates interaction with progress state.

#### Parameters:
- `interaction`: Command interaction
- `message`: Progress message
- `progress`: Progress percentage (0-100)

### showError
```typescript
static async showError(
  interaction: CommandInteraction,
  error: Error,
  ephemeral: boolean = true
): Promise<void>
```

Displays an error message.

#### Parameters:
- `interaction`: Command interaction
- `error`: Error details
- `ephemeral`: Show only to command user

## Integration Examples

### Basic Embed
```typescript
const embed = UIService.createEmbed({
  title: "Search Results",
  description: "Found 5 relevant conversations",
  color: UI_CONSTANTS.COLORS.PRIMARY
});

await interaction.reply({ embeds: [embed] });
```

### Paginated Results
```typescript
const view = await UIService.createPaginatedView(
  conversations,
  {
    itemsPerPage: 5,
    timeout: 300000,
    initialPage: 0
  },
  (items) => createConversationEmbed(items)
);

await interaction.reply(view);
```

### Interactive Buttons
```typescript
const buttons = [
  {
    customId: "prev",
    label: "Previous",
    style: ButtonStyle.Secondary,
    disabled: isFirstPage
  },
  {
    customId: "next",
    label: "Next",
    style: ButtonStyle.Primary,
    disabled: isLastPage
  }
];

const row = UIService.createButtonRow(buttons);
await interaction.reply({ components: [row] });
```

## Error Handling

### Standard Errors
```typescript
try {
  await processCommand(interaction);
} catch (error) {
  await UIService.showError(interaction, error);
}
```

### Progress Updates
```typescript
for (let i = 0; i < items.length; i++) {
  const progress = (i / items.length) * 100;
  await UIService.showProgress(interaction, "Processing...", progress);
  await processItem(items[i]);
}
```

## Best Practices

1. **User Experience**
   - Use consistent styling
   - Provide clear feedback
   - Handle timeouts gracefully

2. **Performance**
   - Optimize embed content
   - Handle large datasets
   - Cache when appropriate

3. **Error Handling**
   - Show user-friendly errors
   - Include recovery options
   - Log detailed errors

## Configuration

```typescript
const UI_CONFIG = {
  // Display settings
  COLORS: {
    PRIMARY: 0x0099ff,
    SUCCESS: 0x00ff00,
    ERROR: 0xff0000,
    WARNING: 0xffff00
  },

  // Timeouts (ms)
  BUTTON_TIMEOUT: 300000,    // 5 minutes
  PROGRESS_INTERVAL: 2000,   // 2 seconds
  
  // Pagination
  DEFAULT_PAGE_SIZE: 5,
  MAX_PAGES: 20,
  
  // Content limits
  MAX_EMBED_LENGTH: 4096,
  MAX_FIELD_LENGTH: 1024
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
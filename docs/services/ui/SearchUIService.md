# SearchUIService

The SearchUIService extends ConversationUIService with specialized components for displaying and interacting with search results.

## Core Features

### Search Display
- Result highlighting
- Relevance indicators
- Filter controls
- Sort options

### Search Interaction
- Result filtering
- Sort controls
- Dynamic updates
- Quick actions

## Method Reference

### displaySearchResults
```typescript
static async displaySearchResults(
  interaction: CommandInteraction,
  results: ConversationWithContext[],
  options?: SearchDisplayOptions
): Promise<void>
```

Displays search results with controls.

#### Parameters:
- `interaction`: Discord interaction
- `results`: Search results
- `options`: Display settings
  - `filters`: Active filters
  - `sorting`: Sort configuration
  - `highlighting`: Highlight options
  - `ephemeral`: Private response

### createSearchEmbed
```typescript
static createSearchEmbed(
  result: ConversationWithContext,
  options?: SearchEmbedOptions
): EmbedBuilder
```

Creates a search result embed.

#### Parameters:
- `result`: Search result
- `options`: Embed settings
  - `highlightQuery`: Search query
  - `showScore`: Show relevance
  - `matchContext`: Show match context
  - `color`: Embed color

#### Returns:
Configured search embed

### handleFiltering
```typescript
static async handleFiltering(
  interaction: ButtonInteraction,
  results: ConversationWithContext[],
  options?: FilterOptions
): Promise<void>
```

Handles filter interactions.

#### Parameters:
- `interaction`: Button interaction
- `results`: All search results
- `options`: Filter settings
  - `filters`: Available filters
  - `currentState`: Active filters
  - `timeout`: Control timeout

### updateSearchDisplay
```typescript
static async updateSearchDisplay(
  interaction: CommandInteraction,
  results: ConversationWithContext[],
  options?: SearchUpdateOptions
): Promise<void>
```

Updates search result display.

#### Parameters:
- `interaction`: Command interaction
- `results`: Updated results
- `options`: Update settings
  - `preserveFilters`: Keep filters
  - `updateSort`: Change sorting
  - `animate`: Animate changes

## Integration Examples

### Basic Search Display
```typescript
// Display search results
await SearchUIService.displaySearchResults(
  interaction,
  searchResults,
  {
    filters: ['date', 'author', 'channel'],
    sorting: { by: 'relevance', order: 'desc' },
    highlighting: true
  }
);
```

### Custom Result Embed
```typescript
// Create search result embed
const embed = SearchUIService.createSearchEmbed(
  result,
  {
    highlightQuery: searchTerm,
    showScore: true,
    matchContext: true
  }
);

await interaction.reply({ embeds: [embed] });
```

### Filter Management
```typescript
// Handle filter changes
await SearchUIService.handleFiltering(
  buttonInteraction,
  allResults,
  {
    filters: {
      date: ['today', 'week', 'month'],
      author: authors,
      channel: channels
    },
    currentState: activeFilters
  }
);
```

## Error Handling

### Display Errors
```typescript
try {
  await SearchUIService.displaySearchResults(interaction, results);
} catch (error) {
  if (error instanceof SearchDisplayError) {
    await interaction.reply({
      content: `Search display error: ${error.message}`,
      ephemeral: true
    });
    return;
  }
  throw error;
}
```

### Filter Errors
```typescript
try {
  await SearchUIService.handleFiltering(interaction, results, options);
} catch (error) {
  if (error instanceof FilterError) {
    await interaction.update({
      components: generateErrorComponents(error)
    });
  }
}
```

## Best Practices

1. **Result Presentation**
   - Clear relevance indicators
   - Context highlighting
   - Intuitive filtering

2. **Performance**
   - Efficient result filtering
   - Smooth updates
   - Responsive controls

3. **User Experience**
   - Save filter preferences
   - Quick result navigation
   - Clear feedback

## Configuration

```typescript
const SEARCH_UI_CONFIG = {
  // Display settings
  RESULTS_PER_PAGE: 5,
  MAX_PAGES: 20,
  CONTROL_TIMEOUT: 300000,
  
  // Result formatting
  HIGHLIGHT_COLOR: 0xffff00,
  MAX_CONTEXT_LENGTH: 200,
  TRUNCATION_MARKER: '...',
  
  // Filter settings
  DEFAULT_FILTERS: ['date', 'author'],
  FILTER_TIMEOUT: 60000,
  MAX_FILTER_OPTIONS: 25,
  
  // Sort settings
  SORT_OPTIONS: [
    'relevance',
    'date',
    'author',
    'length'
  ],
  DEFAULT_SORT: 'relevance'
};
```

For implementation examples, see the [Examples Guide](../../guides/Examples.md).
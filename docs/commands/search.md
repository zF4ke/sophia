# Search Command

The search command allows users to find relevant conversations in a channel using AI-powered analysis.

## Usage

```
/search [topic] [channel] [limit] [include_bots] [ephemeral]
```

## Parameters

### Required
- `topic`: The topic or query to search for
- `channel`: The channel to search in

### Optional
- `limit`: Maximum number of messages to fetch (default: 2000)
- `include_bots`: Whether to include bot messages (default: false)
- `ephemeral`: Whether to show results only to you (default: false)

## Examples

### Basic Search
```
/search topic:project deadlines channel:#team-updates
```

### Advanced Options
```
/search topic:meeting notes channel:#general limit:5000 include_bots:true ephemeral:true
```

### Topic Examples
- "typescript errors"
- "project planning meeting"
- "deployment issues"
- "feature requests"

## Permissions

- Requires READ_MESSAGES permission in target channel
- Optional: Manage Messages for non-ephemeral results
- Rate limited to 10 uses per hour per user

## Response Format

### Success
```typescript
{
  embeds: [
    {
      title: "Search Results",
      description: "Found {count} relevant conversations",
      fields: [
        // Conversation snippets with context
      ],
      footer: "Page {current} of {total}"
    }
  ],
  components: [
    // Navigation buttons
  ]
}
```

### Error States
- Channel not found
- No matching results
- Rate limit exceeded
- Permission denied

## Integration

### With Context Command
```
/search topic:"previous discussion" channel:#project
/context prompt:"summarize these findings" channel:#project
```

### With Cache Command
```
/search topic:"important updates" channel:#announcements
/cache clear channel:#announcements
```

## Best Practices

1. **Search Queries**
   - Use specific topics
   - Include key terms
   - Consider time context

2. **Performance**
   - Use reasonable limits
   - Enable ephemeral for large results
   - Cache frequently searched channels

3. **Result Management**
   - Review all pages
   - Use filters when available
   - Save important results

## Configuration

```typescript
const SEARCH_CONFIG = {
  // Limits
  DEFAULT_LIMIT: 2000,
  MAX_LIMIT: 10000,
  MIN_RELEVANCE: 0.6,
  
  // Display
  RESULTS_PER_PAGE: 5,
  MAX_PAGES: 20,
  
  // Rate Limiting
  MAX_USES_PER_HOUR: 10,
  COOLDOWN: '1m'
};
```

For implementation details, see the [API Documentation](../API.md).
# Context Command

The context command uses channel messages as context for generating AI responses to user queries.

## Usage

```
/context [prompt] [channel] [limit] [include_bots] [ephemeral]
```

## Parameters

### Required
- `prompt`: Your question or instruction for the AI
- `channel`: The channel to use as context

### Optional
- `limit`: Maximum messages to use as context (default: 1000)
- `include_bots`: Include bot messages in context (default: false)
- `ephemeral`: Show response only to you (default: true)

## Examples

### Basic Query
```
/context prompt:"What was decided about the database migration?" channel:#tech-decisions
```

### Advanced Options
```
/context prompt:"Summarize the project timeline" channel:#project-updates limit:2000 include_bots:true ephemeral:false
```

### Prompt Examples
- "What are the main discussion points?"
- "Summarize the key decisions"
- "What are the action items?"
- "When is the next milestone?"

## Permissions

- Requires READ_MESSAGES in target channel
- Optional: SEND_MESSAGES for non-ephemeral responses
- Rate limited to 20 uses per hour per user

## Response Format

### Success
```typescript
{
  embeds: [
    {
      title: "AI Response",
      description: "Based on channel context",
      fields: [
        {
          name: "Your Question",
          value: "{prompt}"
        },
        {
          name: "Response",
          value: "{ai_response}"
        }
      ],
      footer: "Using {message_count} messages as context"
    }
  ]
}
```

### Error States
- Channel not found
- Insufficient context
- Rate limit exceeded
- AI service unavailable

## Integration

### With Search Command
```
// Find relevant discussions
/search topic:"database schema" channel:#tech
// Get detailed analysis
/context prompt:"Explain the schema changes" channel:#tech
```

### With Cache Command
```
// Clear outdated cache
/cache clear channel:#tech
// Get fresh context
/context prompt:"Latest updates?" channel:#tech
```

## Best Practices

1. **Prompt Writing**
   - Be specific and clear
   - Focus on key information
   - Consider time context
   - Use natural language

2. **Context Management**
   - Use appropriate limits
   - Consider message relevance
   - Include necessary bots
   - Update cache if needed

3. **Response Handling**
   - Verify information
   - Save important responses
   - Follow up if needed
   - Share when valuable

## Configuration

```typescript
const CONTEXT_CONFIG = {
  // Context limits
  DEFAULT_LIMIT: 1000,
  MAX_LIMIT: 5000,
  MIN_MESSAGES: 10,
  
  // AI settings
  TEMPERATURE: 0.7,
  MAX_TOKENS: 2000,
  TOP_P: 0.95,
  
  // Rate limiting
  MAX_USES_PER_HOUR: 20,
  COOLDOWN: '30s',
  
  // Display
  MAX_RESPONSE_LENGTH: 2000,
  TRUNCATION_MARKER: '...'
};
```

For implementation details, see the [API Documentation](../API.md).
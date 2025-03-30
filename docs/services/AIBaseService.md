[Back to Index](../API.md)

# AIBaseService

The AIBaseService provides core AI functionality and common utilities for Gemini API integration.

## Core Features

### API Integration
- Gemini API management
- Token handling
- Rate limiting
- Error recovery

### Common Utilities
- Token counting
- Request formatting
- Response parsing
- Error handling

## Method Reference

### initializeAPI
```typescript
static async initializeAPI(
  options?: InitOptions
): Promise<void>
```

Initializes Gemini API client.

#### Parameters:
- `options`: Initialization settings
  - `apiKey`: API key override
  - `timeout`: Request timeout
  - `retries`: Max retries
  - `model`: Model version

### formatPrompt
```typescript
static formatPrompt(
  text: string,
  context: string,
  options?: PromptOptions
): string
```

Formats prompt for AI processing.

#### Parameters:
- `text`: Main prompt text
- `context`: Additional context
- `options`: Format settings
  - `temperature`: Response randomness
  - `maxTokens`: Token limit
  - `template`: Prompt template

#### Returns:
Formatted prompt string

### countTokens
```typescript
static countTokens(
  text: string,
  options?: TokenOptions
): number
```

Estimates token count for text.

#### Parameters:
- `text`: Input text
- `options`: Count settings
  - `model`: Model version
  - `detailed`: Return details
  - `encoding`: Token encoding

#### Returns:
Estimated token count

### handleRateLimit
```typescript
static async handleRateLimit(
  error: Error,
  options?: RetryOptions
): Promise<void>
```

Handles API rate limiting.

#### Parameters:
- `error`: Rate limit error
- `options`: Retry settings
  - `maxRetries`: Maximum attempts
  - `delay`: Retry delay
  - `backoff`: Delay multiplier

## Integration Examples

### Basic API Usage
```typescript
// Initialize API
await AIBaseService.initializeAPI({
  timeout: 30000,
  retries: 3
});

// Format and validate prompt
const prompt = AIBaseService.formatPrompt(
  userQuery,
  context,
  { temperature: 0.7 }
);
```

### Token Management
```typescript
// Check token count
const tokens = AIBaseService.countTokens(text);

if (tokens > MAX_TOKENS) {
  text = await optimizeText(text, MAX_TOKENS);
}
```

### Rate Limit Handling
```typescript
try {
  await makeAPIRequest();
} catch (error) {
  if (error.code === 'RATE_LIMIT') {
    await AIBaseService.handleRateLimit(error, {
      maxRetries: 3,
      delay: 1000
    });
    await makeAPIRequest();
  }
}
```

## Error Handling

### API Errors
```typescript
try {
  await AIBaseService.makeRequest(prompt);
} catch (error) {
  if (error instanceof AIError) {
    switch (error.code) {
      case 'INVALID_API_KEY':
        await handleAuthError(error);
        break;
      case 'QUOTA_EXCEEDED':
        await notifyQuotaExceeded();
        break;
      default:
        throw error;
    }
  }
}
```

### Recovery Strategies
```typescript
static async withRetry<T>(
  operation: () => Promise<T>,
  options: RetryOptions
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (shouldRetry(error, options)) {
      await delay(options.delay);
      return await this.withRetry(operation, {
        ...options,
        retries: options.retries - 1
      });
    }
    throw error;
  }
}
```

## Best Practices

1. **API Management**
   - Handle rate limits
   - Implement retries
   - Monitor quotas

2. **Token Usage**
   - Count tokens accurately
   - Optimize prompts
   - Respect limits

3. **Error Recovery**
   - Graceful degradation
   - Clear error messages
   - Proper logging

## Configuration

```typescript
const AI_BASE_CONFIG = {
  // API settings
  API: {
    TIMEOUT: 30000,
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000,
    DEFAULT_MODEL: 'gemini-pro'
  },
  
  // Token limits
  TOKENS: {
    MAX_PROMPT: 4000,
    MAX_CONTEXT: 8000,
    SAFETY_MARGIN: 100
  },
  
  // Rate limits
  RATE_LIMITS: {
    REQUESTS_PER_MIN: 60,
    TOKENS_PER_MIN: 40000,
    MAX_PARALLEL: 5
  },
  
  // Error handling
  ERRORS: {
    MAX_RETRIES: 3,
    BASE_DELAY: 1000,
    MAX_DELAY: 10000
  }
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
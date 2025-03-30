[Back to Index](../API.md)

# ResponseGenerationService

The ResponseGenerationService handles AI response generation and formatting using the Gemini API.

## Core Features

### Response Generation
- Context-aware responses
- Template-based generation
- Response formatting
- Stream processing

### Template Management
- Response templates
- Dynamic formatting
- Variable interpolation
- Style customization

## Method Reference

### generateResponse
```typescript
static async generateResponse(
  prompt: string,
  context: string,
  options?: GenerationOptions
): Promise<string>
```

Generates an AI response with context.

#### Parameters:
- `prompt`: User query/instruction
- `context`: Conversation context
- `options`: Generation settings
  - `temperature`: Response randomness
  - `maxTokens`: Length limit
  - `format`: Output format
  - `stream`: Enable streaming

#### Returns:
Generated response string

### applyTemplate
```typescript
static applyTemplate(
  template: string,
  variables: Record<string, string>,
  options?: TemplateOptions
): string
```

Applies variables to a response template.

#### Parameters:
- `template`: Response template
- `variables`: Template variables
- `options`: Template settings
  - `escapeHtml`: HTML escaping
  - `fallback`: Default values
  - `formatter`: Custom formatting

#### Returns:
Formatted response string

### streamResponse
```typescript
static async *streamResponse(
  prompt: string,
  context: string,
  options?: StreamOptions
): AsyncGenerator<string>
```

Streams AI response generation.

#### Parameters:
- `prompt`: Generation prompt
- `context`: Background context
- `options`: Stream settings
  - `chunkSize`: Token chunk size
  - `delimiter`: Chunk delimiter
  - `maxDuration`: Stream timeout

#### Returns:
AsyncGenerator of response chunks

### formatResponse
```typescript
static formatResponse(
  response: string,
  options?: FormatOptions
): string
```

Formats a response for display.

#### Parameters:
- `response`: Raw response
- `options`: Format options
  - `style`: Display style
  - `markdown`: Enable markdown
  - `maxLength`: Length limit
  - `truncation`: Truncation style

#### Returns:
Formatted response string

## Integration Examples

### Basic Response Generation
```typescript
// Generate response with context
const response = await ResponseGenerationService.generateResponse(
  userQuery,
  channelContext,
  {
    temperature: 0.7,
    maxTokens: 2000
  }
);

// Format and send
const formatted = ResponseGenerationService.formatResponse(response, {
  style: 'markdown',
  maxLength: 1500
});
```

### Template Usage
```typescript
// Apply template variables
const filled = ResponseGenerationService.applyTemplate(
  responseTemplate,
  {
    username: user.name,
    query: userQuery,
    context: summary
  },
  { escapeHtml: true }
);
```

### Streaming Response
```typescript
// Stream response generation
for await (const chunk of ResponseGenerationService.streamResponse(
  prompt,
  context,
  { chunkSize: 100 }
)) {
  await updateResponse(chunk);
}
```

## Error Handling

### Generation Errors
```typescript
try {
  const response = await ResponseGenerationService.generateResponse(prompt, context);
} catch (error) {
  if (error instanceof GenerationError) {
    console.error('Generation failed:', error.message);
    return getFallbackResponse(prompt);
  }
  throw error;
}
```

### Stream Handling
```typescript
try {
  for await (const chunk of ResponseGenerationService.streamResponse(prompt, context)) {
    await processChunk(chunk);
  }
} catch (error) {
  console.warn('Stream interrupted:', error);
  await finalizeResponse();
}
```

## Best Practices

1. **Response Quality**
   - Provide sufficient context
   - Use appropriate temperature
   - Validate outputs

2. **Performance**
   - Optimize context size
   - Stream long responses
   - Cache templates

3. **User Experience**
   - Handle interruptions
   - Show progress
   - Format consistently

## Configuration

```typescript
const RESPONSE_CONFIG = {
  // Generation settings
  DEFAULT_TEMPERATURE: 0.7,
  MAX_TOKENS: 4000,
  MIN_TOKENS: 100,
  
  // Stream settings
  CHUNK_SIZE: 100,
  STREAM_TIMEOUT: 30000,
  MAX_CHUNKS: 50,
  
  // Format settings
  MAX_LENGTH: 2000,
  TRUNCATION_MARKER: '...',
  DEFAULT_STYLE: 'markdown',
  
  // Template settings
  VARIABLE_PATTERN: /\${(\w+)}/g,
  DEFAULT_FALLBACK: '',
  ESCAPE_HTML: true
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
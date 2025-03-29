[Back to Index](../API.md)

# ResponseGenerationService

The ResponseGenerationService is responsible for generating AI responses based on user prompts and contexts. It provides various methods for different types of response generation needs using Google's Generative AI.

## Methods

### `generateContextualResponse`

```typescript
public static async generateContextualResponse(
  prompt: string, 
  context: string,
  additionalInstructions: string = ""
): Promise<string>
```

Generates a contextual response to a user prompt using provided conversation context.

#### Features:
- Uses conversation context to inform the AI response
- Supports additional instructions to fine-tune response
- Uses optimized context model for balanced output
- Includes error handling

#### Parameters:
- `prompt: string` - User query or instruction
- `context: string` - Conversation context to inform the response
- `additionalInstructions: string` - Optional additional instructions for the AI

#### Returns:
- `Promise<string>` - The generated response

#### Example:
```typescript
const response = await ResponseGenerationService.generateContextualResponse(
  "What was the conclusion about the project timeline?",
  conversationContext,
  "Focus on specific dates mentioned"
);
```

### `generateCustomResponse`

```typescript
public static async generateCustomResponse(
  promptTemplate: string, 
  params: Record<string, string>,
  temperature: number = 0.4
): Promise<string>
```

Generates a response using a custom prompt template and specified parameters.

#### Features:
- Template-based prompt generation with parameter substitution
- Adjustable temperature for controlling response randomness
- Custom model configuration for specific use cases
- Error handling with informative errors

#### Parameters:
- `promptTemplate: string` - Template string with placeholders in `{{parameter}}` format
- `params: Record<string, string>` - Object containing values to replace placeholders
- `temperature: number` - Temperature parameter (0.0-1.0) for controlling response randomness

#### Returns:
- `Promise<string>` - The generated response

#### Example:
```typescript
const response = await ResponseGenerationService.generateCustomResponse(
  "Write a {{tone}} explanation of {{concept}} suitable for {{audience}}",
  {
    tone: "friendly",
    concept: "artificial intelligence",
    audience: "beginners"
  },
  0.6
);
```

### `generateSummary`

```typescript
public static async generateSummary(
  text: string, 
  maxLength: number = 500,
  focusTopics: string[] = []
): Promise<string>
```

Summarizes a long text or conversation.

#### Features:
- Creates concise summaries with length control
- Optional topic focus for targeted summaries
- Uses contextual model for coherent output
- Error handling with fallback messaging

#### Parameters:
- `text: string` - Text to be summarized
- `maxLength: number` - Target maximum length for summary (default: 500 characters)
- `focusTopics: string[]` - Optional array of topics to focus on in the summary

#### Returns:
- `Promise<string>` - The generated summary

#### Example:
```typescript
const longDiscussion = getConversationText(conversation);
const summary = await ResponseGenerationService.generateSummary(
  longDiscussion,
  300,
  ["budget constraints", "timeline"]
);
```

### `generateConversationResponse`

```typescript
public static async generateConversationResponse(
  previousMessages: {role: string, content: string}[],
  newUserInput: string
): Promise<string>
```

Generates a follow-up response based on previous conversation and new input.

#### Features:
- Maintains conversation continuity and context
- Handles multi-turn conversations naturally
- Formats conversation history appropriately for the AI
- Error handling with default responses

#### Parameters:
- `previousMessages: {role: string, content: string}[]` - Array of previous message pairs
- `newUserInput: string` - Latest user input to respond to

#### Returns:
- `Promise<string>` - The follow-up response

#### Example:
```typescript
const conversationHistory = [
  { role: "user", content: "Can you explain Docker containers?" },
  { role: "assistant", content: "Docker containers are lightweight, standalone executable packages..." }
];

const response = await ResponseGenerationService.generateConversationResponse(
  conversationHistory,
  "How do they differ from virtual machines?"
);
```

## Usage Recommendations

- Use `generateContextualResponse` when you have specific conversation context to inform responses
- Use `generateCustomResponse` for template-based generation with variable parameters
- Use `generateSummary` for condensing long texts or conversations
- Use `generateConversationResponse` for maintaining conversational continuity
- Adjust temperature based on need for creativity vs. determinism:
  - Lower values (0.2-0.4) for factual, consistent responses
  - Medium values (0.4-0.7) for balanced responses
  - Higher values (0.7-1.0) for more creative, varied responses
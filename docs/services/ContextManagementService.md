[Back to Index](../API.md)

# ContextManagementService

The ContextManagementService is responsible for managing conversation contexts for AI interactions. It provides functionality for selecting, formatting, and optimizing conversation contexts for use in AI prompts.

## Properties

### Configuration Constants

```typescript
private static readonly DEFAULT_MAX_CONTEXT_CHARS = 50000;
```

The default maximum context length in characters (50,000).

## Methods

### `selectConversationsForContext`

```typescript
public static selectConversationsForContext(
  conversations: Message[][], 
  prompt: string, 
  maxChars: number = this.DEFAULT_MAX_CONTEXT_CHARS
): Message[][]
```

Selects the most relevant conversations for providing context to AI.

#### Features:
- Uses keyword matching to determine relevance to the prompt
- Scores conversations based on keyword matches and conversation complexity
- Ensures total context stays within character limits
- Prioritizes more relevant conversations

#### Parameters:
- `conversations: Message[][]` - Array of conversations to select from
- `prompt: string` - User prompt to base relevance on
- `maxChars: number` - Maximum character limit for context (default: 50000)

#### Returns:
- `Message[][]` - Array of selected conversations within character limit

#### Example:
```typescript
const relevantContext = ContextManagementService.selectConversationsForContext(
  allConversations,
  "project deadline",
  30000
);
```

### `formatConversationsAsContext`

```typescript
public static formatConversationsAsContext(conversations: Message[][]): string
```

Formats conversations into a structured context string for AI.

#### Features:
- Creates a clear, numbered format for conversations
- Preserves author information with each message
- Organizes content in a way that's optimal for AI processing

#### Parameters:
- `conversations: Message[][]` - Array of conversations to format

#### Returns:
- `string` - Formatted context string

#### Example:
```typescript
const selectedContexts = ContextManagementService.selectConversationsForContext(
  allConversations, 
  userQuery
);
const formattedContext = ContextManagementService.formatConversationsAsContext(selectedContexts);
```

### `createContextualPrompt`

```typescript
public static createContextualPrompt(
  userPrompt: string, 
  context: string,
  additionalInstructions: string = ""
): string
```

Creates a structured prompt with context for AI response generation.

#### Features:
- Adds clear section headers for context and user prompt
- Includes guidelines for how the AI should use the context
- Supports additional customization through instructions parameter

#### Parameters:
- `userPrompt: string` - The user's original query or request
- `context: string` - Context information from conversations
- `additionalInstructions: string` - Any additional instructions for AI behavior

#### Returns:
- `string` - Complete formatted prompt with context

#### Example:
```typescript
const prompt = ContextManagementService.createContextualPrompt(
  "What was the final decision about the project timeline?",
  formattedContext,
  "Focus on the most recent information and concrete dates mentioned."
);
```

### `optimizeContextForTokenLimit`

```typescript
public static optimizeContextForTokenLimit(
  conversations: Message[][], 
  prompt: string, 
  maxTokens: number = 8000
): string
```

Optimizes context by focusing on most relevant parts if context exceeds token limits.

#### Features:
- Approximates token count from character length
- Intelligently reduces context while preserving most relevant information
- Prioritizes conversations and messages with keyword matches
- Can extract portions of conversations if needed

#### Parameters:
- `conversations: Message[][]` - Array of conversations
- `prompt: string` - User prompt
- `maxTokens: number` - Approximate maximum token count for context

#### Returns:
- `string` - Optimized context string

#### Example:
```typescript
// When working with a model with strict token limits
const optimizedContext = ContextManagementService.optimizeContextForTokenLimit(
  allConversations,
  userQuery,
  4000 // Limit for smaller models
);
```

## Usage Recommendations

- Use `selectConversationsForContext` to intelligently choose which conversations to include
- Always format conversations with `formatConversationsAsContext` for consistency
- Use `createContextualPrompt` to create well-structured prompts for AI
- Consider token limits of the AI model and use `optimizeContextForTokenLimit` when necessary
- For very large conversation sets, apply pre-filtering before selection
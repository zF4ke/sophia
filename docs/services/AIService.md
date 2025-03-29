[Back to Index](../API.md)

# AIService

The AIService is the main entry point for AI-related operations. It coordinates specialized AI services to provide a comprehensive interface for conversation analysis, context management, and response generation using Google's Generative AI (Gemini).

## Overview

AIService extends AIBaseService and serves as a facade for the AI subsystem. It provides high-level methods while delegating specialized operations to dedicated services:

- **ConversationAnalysisService** - For analyzing conversations and determining relevance
- **ContextManagementService** - For managing conversation contexts
- **ResponseGenerationService** - For generating AI responses
- **TextProcessingService** - For text processing operations

## Methods

### `analyzeConversations`

```typescript
static async analyzeConversations(
  conversations: Message[][],
  topic: string,
  interaction: ChatInputCommandInteraction
): Promise<ConversationWithContext[]>
```

Analyzes conversations for relevance to a specific topic using Gemini AI.

#### Features:
- Delegates to ConversationAnalysisService for AI-powered analysis
- Progress updates during analysis
- Relevance scoring for conversations

#### Parameters:
- `conversations: Message[][]` - Array of message arrays representing conversations
- `topic: string` - The search topic to analyze against
- `interaction: ChatInputCommandInteraction` - Discord interaction for progress updates

#### Returns:
- `Promise<ConversationWithContext[]>` - Array of relevant conversations with context

#### Example:
```typescript
const relevantConversations = await AIService.analyzeConversations(
  validConversations, 
  "machine learning", 
  interaction
);
console.log(`Found ${relevantConversations.length} relevant conversations`);
```

### `fallbackKeywordSearch`

```typescript
static fallbackKeywordSearch(
  conversations: Message[][], 
  topic: string
): ConversationWithContext[]
```

Provides basic keyword search when AI analysis fails.

#### Features:
- Graceful degradation when AI service is unavailable
- Basic relevance scoring based on keyword matches
- Maintains consistent return format for seamless integration

#### Parameters:
- `conversations: Message[][]` - Array of conversation groups
- `topic: string` - Search topic

#### Returns:
- `ConversationWithContext[]` - Array of conversations containing the topic keyword

#### Example:
```typescript
try {
  return await AIService.analyzeConversations(conversations, topic, interaction);
} catch (error) {
  console.error('AI analysis failed, falling back to keyword search:', error);
  return AIService.fallbackKeywordSearch(conversations, topic);
}
```

### `extractKeywords`

```typescript
static extractKeywords(text: string): string[]
```

Extracts meaningful keywords from text by removing common words and short terms.

#### Features:
- Delegates to TextProcessingService
- Filters out common stopwords in multiple languages
- Removes very short terms

#### Parameters:
- `text: string` - The input text to extract keywords from

#### Returns:
- `string[]` - Array of extracted keywords

#### Example:
```typescript
const keywords = AIService.extractKeywords("How to implement machine learning models in JavaScript");
// Returns: ["implement", "machine", "learning", "models", "javascript"]
```

### `selectConversationsForContext`

```typescript
static selectConversationsForContext(
  conversations: Message[][], 
  prompt: string, 
  maxChars: number = 50000
): Message[][]
```

Selects the most relevant conversations for providing context to AI.

#### Features:
- Delegates to ContextManagementService
- Selects conversations based on relevance to prompt
- Respects maximum character limit for context

#### Parameters:
- `conversations: Message[][]` - Array of conversations to select from
- `prompt: string` - User prompt to match against
- `maxChars: number` - Maximum character limit for context (default: 50000)

#### Returns:
- `Message[][]` - Selected conversations within character limit

### `formatConversationsAsContext`

```typescript
static formatConversationsAsContext(conversations: Message[][]): string
```

Formats conversations into a structured context string for AI.

#### Features:
- Delegates to ContextManagementService
- Creates readable conversation format for AI consumption
- Preserves username and message content

#### Parameters:
- `conversations: Message[][]` - Array of conversations to format

#### Returns:
- `string` - Formatted context string

### `generateContextualResponse`

```typescript
static async generateContextualResponse(
  prompt: string, 
  context: string
): Promise<string>
```

Generates a contextual response to a user prompt using provided conversation context.

#### Features:
- Delegates to ResponseGenerationService
- Creates AI responses informed by conversational context
- Handles errors gracefully

#### Parameters:
- `prompt: string` - User query or instruction
- `context: string` - Conversation context to inform the response

#### Returns:
- `Promise<string>` - Generated AI response

#### Example:
```typescript
const context = AIService.formatConversationsAsContext(selectedConversations);
const response = await AIService.generateContextualResponse(
  "What was decided about the project deadline?", 
  context
);
```

### `optimizeContextForTokenLimit`

```typescript
static optimizeContextForTokenLimit(
  conversations: Message[][], 
  prompt: string, 
  maxTokens: number = 8000
): string
```

Optimizes context for token limitations while preserving relevance.

#### Features:
- Delegates to ContextManagementService
- Intelligently trims context to stay within token limits
- Prioritizes most relevant conversations and messages

#### Parameters:
- `conversations: Message[][]` - Array of conversations
- `prompt: string` - User prompt
- `maxTokens: number` - Approximate maximum token count for context (default: 8000)

#### Returns:
- `string` - Optimized context string

### `generateSummary`

```typescript
static async generateSummary(
  text: string, 
  maxLength: number = 500,
  focusTopics: string[] = []
): Promise<string>
```

Summarizes a long text or conversation.

#### Features:
- Delegates to ResponseGenerationService
- Creates concise summaries of longer texts
- Optional focus on specific topics

#### Parameters:
- `text: string` - Text to be summarized
- `maxLength: number` - Target maximum length for summary (default: 500)
- `focusTopics: string[]` - Optional array of topics to focus on

#### Returns:
- `Promise<string>` - Generated summary

### `generateCustomResponse`

```typescript
static async generateCustomResponse(
  promptTemplate: string, 
  params: Record<string, string>,
  temperature: number = 0.4
): Promise<string>
```

Generates a response using a custom prompt template and specified parameters.

#### Features:
- Delegates to ResponseGenerationService
- Supports template-based prompt generation
- Customizable temperature parameter for response randomness

#### Parameters:
- `promptTemplate: string` - Template string with placeholders
- `params: Record<string, string>` - Object containing values to replace placeholders
- `temperature: number` - Temperature parameter for controlling response randomness (0.0-1.0)

#### Returns:
- `Promise<string>` - Generated response

#### Example:
```typescript
const response = await AIService.generateCustomResponse(
  "Write a {{style}} explanation of {{topic}} for {{audience}}",
  {
    style: "simple",
    topic: "quantum computing",
    audience: "beginners"
  },
  0.7
);
```

### `cleanText`

```typescript
static cleanText(text: string): string
```

Cleans text by removing Markdown formatting characters.

#### Features:
- Delegates to TextProcessingService
- Removes markdown characters for cleaner AI processing
- Preserves core content

#### Parameters:
- `text: string` - The text to clean

#### Returns:
- `string` - Cleaned text without markdown characters

## Best Practices

- Use AIService as the main entry point for AI operations
- Handle potential AI failures gracefully with fallbackKeywordSearch
- Optimize context for token limits in larger conversations
- Use the specialized AI services directly only when requiring very specific functionality
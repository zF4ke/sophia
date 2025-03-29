[Back to Index](../API.md)

# ConversationAnalysisService

The ConversationAnalysisService specializes in analyzing conversations and determining their relevance to specific topics using AI. It provides sophisticated conversation scoring and ranking capabilities.

## Methods

### `analyzeConversations`

```typescript
public static async analyzeConversations(
  conversations: Message[][],
  topic: string,
  interaction: ChatInputCommandInteraction
): Promise<ConversationWithContext[]>
```

Analyzes conversations for relevance to a specific topic.

#### Features:
- Uses AI to analyze conversation relevance to topics
- Implements batch processing for efficient API usage
- Provides progress updates during analysis
- Applies multiple relevance validation rules
- Sorts results by relevance score

#### Parameters:
- `conversations: Message[][]` - Array of message arrays representing conversations
- `topic: string` - The topic to analyze relevance against
- `interaction: ChatInputCommandInteraction` - Discord interaction for updating status

#### Returns:
- `Promise<ConversationWithContext[]>` - Array of relevant conversations with context and relevance scores

#### Example:
```typescript
const relevantConversations = await ConversationAnalysisService.analyzeConversations(
  validConversations, 
  "project planning", 
  interaction
);
```

### `calculateBatchSize` (Private)

```typescript
private static calculateBatchSize(totalConversations: number): number
```

Calculates optimal batch size based on total number of conversations.

#### Features:
- Dynamically adjusts batch size based on dataset size
- Optimizes for both API efficiency and response time
- Scales from small to very large datasets

#### Parameters:
- `totalConversations: number` - Total number of conversations to analyze

#### Returns:
- `number` - Appropriate batch size for processing

### `validateRelevance` (Private)

```typescript
private static validateRelevance(
  conversation: ConversationWithContext, 
  topic: string
): boolean
```

Validates whether a conversation is truly relevant to the topic using multiple criteria.

#### Features:
- Applies tiered validation based on different relevance measures
- Checks for direct topic mentions
- Evaluates keyword matches
- Considers conversation engagement level
- Uses relevance score thresholds

#### Parameters:
- `conversation: ConversationWithContext` - Conversation with context and initial relevance score
- `topic: string` - Topic to validate relevance against

#### Returns:
- `boolean` - Boolean indicating if conversation passes relevance checks

### `analyzeBatch` (Private)

```typescript
private static async analyzeBatch(
  conversations: Message[][], 
  topic: string,
  topicKeywords: string[] 
): Promise<AIAnalysisResult[]>
```

Analyzes a batch of conversations using AI to determine relevance to a topic.

#### Features:
- Creates AI-optimized conversation formatting
- Implements detailed relevance scoring criteria
- Processes multiple conversations in a single AI request
- Parses AI response to extract structured scores
- Includes error handling with fallback analysis

#### Parameters:
- `conversations: Message[][]` - Batch of conversations to analyze
- `topic: string` - Topic to analyze against
- `topicKeywords: string[]` - Pre-extracted keywords from the topic

#### Returns:
- `Promise<AIAnalysisResult[]>` - Array of analysis results with relevance scores

### `fallbackAnalysis` (Private)

```typescript
private static fallbackAnalysis(
  conversation: Message[], 
  topic: string, 
  topicKeywords: string[]
): AIAnalysisResult
```

Performs fallback analysis when AI analysis fails.

#### Features:
- Keyword-based relevance scoring
- Direct topic mention detection
- Content quality assessment
- Conversation engagement evaluation
- Multi-factor relevance determination

#### Parameters:
- `conversation: Message[]` - The conversation to analyze
- `topic: string` - Topic to analyze against
- `topicKeywords: string[]` - Pre-extracted keywords from the topic

#### Returns:
- `AIAnalysisResult` - Analysis result with relevance score

### `fallbackKeywordSearch`

```typescript
public static fallbackKeywordSearch(
  conversations: Message[][], 
  topic: string
): ConversationWithContext[]
```

Performs keyword-based search across conversations when AI search fails.

#### Features:
- Analyzes all conversations using keyword matching
- Applies relevance scoring similar to AI scoring
- Filters results based on minimum content and relevance thresholds
- Sorts results by relevance score

#### Parameters:
- `conversations: Message[][]` - Array of conversations to search through
- `topic: string` - Topic to search for

#### Returns:
- `ConversationWithContext[]` - Array of relevant conversations with context and scores

#### Example:
```typescript
try {
  return await ConversationAnalysisService.analyzeConversations(conversations, topic, interaction);
} catch (error) {
  console.error('AI analysis failed, falling back to keyword search:', error);
  return ConversationAnalysisService.fallbackKeywordSearch(conversations, topic);
}
```

## Scoring System

The service uses a multi-factor scoring system (0-10 scale) based on:

1. **Direct Topic Match (0-4 points)**
   - Exact topic mentions: 4 points
   - Key topic terms: 2-3 points
   - Topic synonyms/related terms: 1-2 points

2. **Context Relevance (0-3 points)**
   - Direct topic discussion: 3 points
   - Related concepts/context: 1-2 points
   - Implicit references: 1 point

3. **Conversation Quality (0-3 points)**
   - Meaningful discussion: 2-3 points
   - Multiple messages: 1-2 points
   - Information value: 1 point

Conversations typically need a score of 3 or higher to be considered relevant.

## Best Practices

- Always implement error handling around AI analysis
- Use the fallback keyword search as a reliable backup
- Consider preprocessing conversations to filter out obvious irrelevant content
- Adjust batch sizes for your specific application performance needs
- Process conversations in smaller batches for real-time applications
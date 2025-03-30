[Back to Index](../API.md)

# ConversationAnalysisService

The ConversationAnalysisService provides advanced conversation analysis and understanding using Gemini AI.

## Core Features

### Analysis Capabilities
- Topic relevance analysis
- Sentiment detection
- Intent recognition
- Entity extraction

### Conversation Processing
- Thread analysis
- Context correlation
- Participant tracking
- Time-based analysis

## Method Reference

### analyzeConversation
```typescript
static async analyzeConversation(
  conversation: Message[],
  options?: AnalysisOptions
): Promise<ConversationAnalysis>
```

Performs comprehensive conversation analysis.

#### Parameters:
- `conversation`: Array of messages
- `options`: Analysis settings
  - `features`: Analysis features to include
  - `depth`: Analysis depth level
  - `maxTokens`: Token limit

#### Returns:
Detailed conversation analysis

### detectTopic
```typescript
static async detectTopic(
  conversation: Message[],
  options?: TopicOptions
): Promise<TopicAnalysis>
```

Identifies conversation topics.

#### Parameters:
- `conversation`: Message array
- `options`: Topic detection settings
  - `minConfidence`: Minimum confidence
  - `maxTopics`: Maximum topics
  - `hierarchical`: Enable topic hierarchy

#### Returns:
Topic analysis results

### analyzeSentiment
```typescript
static async analyzeSentiment(
  messages: Message[],
  options?: SentimentOptions
): Promise<SentimentAnalysis>
```

Analyzes conversation sentiment.

#### Parameters:
- `messages`: Messages to analyze
- `options`: Sentiment settings
  - `granularity`: Analysis detail level
  - `aggregation`: Result aggregation
  - `timeline`: Include temporal analysis

#### Returns:
Sentiment analysis results

### extractEntities
```typescript
static async extractEntities(
  conversation: Message[],
  options?: EntityOptions
): Promise<EntityAnalysis>
```

Extracts named entities and concepts.

#### Parameters:
- `conversation`: Message array
- `options`: Extraction settings
  - `types`: Entity types to extract
  - `confidence`: Minimum confidence
  - `resolution`: Enable entity resolution

#### Returns:
Extracted entities and metadata

## Integration Examples

### Basic Analysis
```typescript
// Analyze conversation
const analysis = await ConversationAnalysisService.analyzeConversation(
  messages,
  {
    features: ['topics', 'sentiment', 'entities'],
    depth: 'detailed'
  }
);

// Process results
if (analysis.confidence > 0.8) {
  await processAnalysisResults(analysis);
}
```

### Topic Detection
```typescript
// Detect conversation topics
const topics = await ConversationAnalysisService.detectTopic(messages, {
  minConfidence: 0.7,
  maxTopics: 3,
  hierarchical: true
});

// Handle topics
for (const topic of topics.mainTopics) {
  await trackConversationTopic(topic);
}
```

### Sentiment Analysis
```typescript
// Analyze sentiment over time
const sentiment = await ConversationAnalysisService.analyzeSentiment(
  messages,
  {
    granularity: 'message',
    timeline: true
  }
);

if (sentiment.overall < 0.3) {
  await flagNegativeConversation(conversation);
}
```

## Error Handling

### Analysis Errors
```typescript
try {
  const analysis = await ConversationAnalysisService.analyzeConversation(messages);
} catch (error) {
  if (error instanceof AIAnalysisError) {
    console.error('Analysis failed:', error.message);
    return await performBasicAnalysis(messages);
  }
  throw error;
}
```

### Fallback Handling
```typescript
try {
  return await ConversationAnalysisService.detectTopic(messages);
} catch (error) {
  console.warn('Topic detection failed:', error);
  return {
    confidence: 0,
    topics: [],
    error: error.message
  };
}
```

## Best Practices

1. **Analysis Quality**
   - Set appropriate confidence thresholds
   - Use proper feature combinations
   - Validate results

2. **Performance**
   - Optimize token usage
   - Batch similar analyses
   - Cache frequent results

3. **Integration**
   - Coordinate with other services
   - Handle partial results
   - Implement fallbacks

## Configuration

```typescript
const ANALYSIS_CONFIG = {
  // Analysis settings
  DEFAULT_CONFIDENCE: 0.6,
  MAX_TOPICS: 5,
  SENTIMENT_GRANULARITY: 'message',
  
  // Processing limits
  MAX_MESSAGES: 100,
  MAX_TOKENS: 8000,
  BATCH_SIZE: 20,
  
  // Feature flags
  ENABLE_HIERARCHY: true,
  ENABLE_TIMELINE: true,
  ENABLE_ENTITY_RESOLUTION: true,
  
  // Cache settings
  CACHE_LIFETIME: 3600,
  MAX_CACHE_ENTRIES: 1000
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
[Back to Index](../API.md)

# Type Definitions

This document defines the core types used throughout Sophia3.

## Message Types

### Message
```typescript
interface Message {
  id: string;
  content: string;
  author: User;
  channelId: string;
  guildId: string;
  createdTimestamp: number;
  mentions: MessageMentions;
  attachments: Collection<string, MessageAttachment>;
}
```

### ConversationWithContext
```typescript
interface ConversationWithContext {
  messages: Message[];
  context: {
    before: Message[];
    after: Message[];
  };
  metadata: ConversationMetadata;
}
```

### ConversationMetadata
```typescript
interface ConversationMetadata {
  id: string;
  topic?: string;
  participants: string[];
  startTime: number;
  endTime: number;
  messageCount: number;
  relevanceScore?: number;
}
```

## Analysis Types

### AnalysisOptions
```typescript
interface AnalysisOptions {
  features?: ('topics' | 'sentiment' | 'entities')[];
  depth?: 'basic' | 'detailed';
  maxTokens?: number;
  minConfidence?: number;
}
```

### TopicAnalysis
```typescript
interface TopicAnalysis {
  mainTopics: string[];
  subtopics: Record<string, string[]>;
  confidence: number;
  relevance: number;
}
```

### SentimentAnalysis
```typescript
interface SentimentAnalysis {
  overall: number;  // -1 to 1
  messages: {
    id: string;
    sentiment: number;
    confidence: number;
  }[];
  timeline?: {
    timestamp: number;
    sentiment: number;
  }[];
}
```

## Cache Types

### CacheData
```typescript
interface CacheData {
  messages: Message[];
  metadata: {
    channelId: string;
    lastUpdate: number;
    messageCount: number;
    expiresAt?: number;
  };
  analysis?: {
    topics?: TopicAnalysis;
    sentiment?: SentimentAnalysis;
  };
}
```

### CacheOptions
```typescript
interface CacheOptions {
  compress?: boolean;
  expiry?: string | number;
  priority?: number;
  tags?: string[];
}
```

## UI Types

### DisplayOptions
```typescript
interface DisplayOptions {
  itemsPerPage?: number;
  showMetadata?: boolean;
  format?: 'compact' | 'detailed';
  ephemeral?: boolean;
}
```

### EmbedOptions
```typescript
interface EmbedOptions {
  color?: number;
  showTimestamps?: boolean;
  includeContext?: boolean;
  maxLength?: number;
  template?: string;
}
```

## Security Types

### AccessOptions
```typescript
interface AccessOptions {
  requireModerator?: boolean;
  allowBots?: boolean;
  channels?: string[];
  expiry?: string | number;
}
```

### RateLimitOptions
```typescript
interface RateLimitOptions {
  maxUses: number;
  window: string | number;
  cooldown?: string | number;
}
```

## Error Types

### AIError
```typescript
class AIError extends Error {
  code: string;
  details?: any;
  retryable: boolean;
}
```

### SecurityError
```typescript
class SecurityError extends Error {
  code: string;
  user: string;
  command: string;
}
```

### CacheError
```typescript
class CacheError extends Error {
  code: string;
  channelId?: string;
  cacheKey?: string;
}
```

## AI Types

### PromptOptions
```typescript
interface PromptOptions {
  temperature?: number;
  maxTokens?: number;
  template?: string;
  stream?: boolean;
}
```

### TokenOptions
```typescript
interface TokenOptions {
  model?: string;
  detailed?: boolean;
  encoding?: string;
}
```

## Service Response Types

### ConversationAnalysis
```typescript
interface ConversationAnalysis {
  topics: TopicAnalysis;
  sentiment?: SentimentAnalysis;
  entities?: EntityAnalysis;
  confidence: number;
  metadata: ConversationMetadata;
}
```

### RateLimitResult
```typescript
interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  reset: number;
  retryAfter?: number;
}
```

For usage examples, see the [Examples Guide](../guides/Examples.md).
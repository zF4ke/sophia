[Back to Index](../API.md)

# Constants

This document describes the important constants used throughout the Sophia3 application.

## Emoji Constants

The `EMOJIS` constant provides a consistent set of emojis used across the application UI for various indicators and actions.

```typescript
export const EMOJIS = {
    search: "🔍",
    conversation: "💬",
    page: "📄",
    relevance: "⭐",
    channel: "📌",
    time: "⏱️",
    error: "❌",
    success: "✅",
    warning: "⚠️"
} as const;
```

These emojis are used consistently across the application to indicate:

- **search**: Search operations and results
- **conversation**: Conversation groups
- **page**: Pagination and page navigation
- **relevance**: Relevance scores in search results
- **channel**: Channel references
- **time**: Timestamps and timing information
- **error**: Error messages
- **success**: Success messages and confirmations
- **warning**: Warning messages and alerts

## Discord Platform Constants

The `DISCORD` object contains platform-specific constants related to Discord limitations and requirements.

```typescript
export const DISCORD = {
    /**
     * Maximum character limit for a single Discord message
     * Messages exceeding this limit need to be split
     */
    MESSAGE_LIMIT: 2000
};
```

These constants are used to handle Discord-specific limitations:

- **MESSAGE_LIMIT**: The maximum number of characters allowed in a single Discord message (2000). Used by commands like `ask` and `context` to split long responses into multiple messages when necessary.

## Admin ID Constants

The `ADMIN_IDS` array contains the Discord user IDs of administrators who have elevated permissions.

```typescript
export const ADMIN_IDS = [
    "676156690395037713",
    "111591984245780480"
];
```

These IDs are used for authorization checks in commands like `search` and `context` that are restricted to admin users.

## Response Messages

### Busy Messages

The `busy` array contains randomized response messages used when the bot is unable to process a request.

```typescript
export const busy = [
    "Eae, eu to ocupada agora. Logo mais eu volto.",
    "Oi, eu to ocupada agora. Logo mais eu volto.",
    "Beleza. Olha, agora não da pra eu te ajudar. Logo mais eu volto.",
    "Fala, olha eu to ocupada agora, mas mais tarde eu volto.",
    "Oi! No momento to ocupada, mas depois a gente se fala.",
    // Additional messages...
];
```

These messages are used primarily in the `oi` command to provide friendly responses when the bot is not available.

## System Constants

This document defines the constants and configuration values used throughout Sophia3.

### UI Constants

```typescript
export const UI_CONSTANTS = {
  // Colors
  COLORS: {
    PRIMARY: 0x0099ff,
    SUCCESS: 0x00ff00,
    ERROR: 0xff0000,
    WARNING: 0xffff00,
    INFO: 0x7289da,
    DEFAULT: 0x2f3136
  },

  // Timeouts (ms)
  TIMEOUTS: {
    BUTTON: 300000,         // 5 minutes
    MENU: 120000,          // 2 minutes
    PROGRESS: 2000,        // 2 seconds
    MESSAGE: 15000         // 15 seconds
  },

  // Display Limits
  LIMITS: {
    EMBED_LENGTH: 4096,
    FIELD_LENGTH: 1024,
    TITLE_LENGTH: 256,
    ITEMS_PER_PAGE: 5,
    MAX_FIELDS: 25
  },

  // Buttons
  BUTTONS: {
    PREVIOUS: '⬅️',
    NEXT: '➡️',
    CLOSE: '❌',
    REFRESH: '🔄',
    FILTER: '🔍'
  }
};
```

### Message Constants

```typescript
export const MESSAGE_CONSTANTS = {
  // Fetch Limits
  FETCH: {
    DEFAULT_LIMIT: 2000,
    MAX_LIMIT: 10000,
    BATCH_SIZE: 100,
    FETCH_DELAY: 1000
  },

  // Cache Settings
  CACHE: {
    MAX_SIZE: 10000,
    DEFAULT_EXPIRY: '1h',
    CLEANUP_INTERVAL: '5m',
    COMPRESSION: true
  },

  // Context Window
  CONTEXT: {
    BEFORE_MESSAGES: 5,
    AFTER_MESSAGES: 5,
    MAX_CONTEXT: 8000,
    MIN_CONTEXT: 100
  },

  // Group Settings
  GROUPING: {
    TIME_THRESHOLD: 300000,  // 5 minutes
    MIN_MESSAGES: 2,
    MAX_GAP: 3600000        // 1 hour
  }
};
```

### Security Constants

```typescript
export const SECURITY_CONSTANTS = {
  // Permission Levels
  LEVELS: {
    ADMIN: 3,
    MOD: 2,
    USER: 1,
    RESTRICTED: 0
  },

  // Rate Limits
  RATE_LIMITS: {
    SEARCH: {
      USES: 10,
      WINDOW: '1h',
      COOLDOWN: '1m'
    },
    CONTEXT: {
      USES: 20,
      WINDOW: '1h',
      COOLDOWN: '30s'
    },
    GETMESSAGE: {
      USES: 30,
      WINDOW: '1h',
      COOLDOWN: '10s'
    }
  },

  // Access Control
  ACCESS: {
    DEFAULT_DURATION: '30d',
    MAX_DURATION: '365d',
    MIN_DURATION: '1d',
    LOG_CHANGES: true
  },

  // Command Categories
  CATEGORIES: {
    ADMIN: ['access', 'config'],
    MOD: ['search', 'context'],
    USER: ['getmessage'],
    PUBLIC: ['help', 'ping']
  }
};
```

### AI Constants

```typescript
export const AI_CONSTANTS = {
  // Model Settings
  MODEL: {
    DEFAULT: 'gemini-pro',
    CONTEXT_LENGTH: 8192,
    MAX_OUTPUT: 4096,
    SAFETY_MARGIN: 100
  },

  // Generation Settings
  GENERATION: {
    DEFAULT_TEMP: 0.7,
    TOP_P: 0.95,
    TOP_K: 40,
    MIN_TOKENS: 100
  },

  // Rate Limits
  LIMITS: {
    REQUESTS_PER_MIN: 60,
    TOKENS_PER_MIN: 40000,
    MAX_PARALLEL: 5,
    RETRY_DELAY: 1000
  },

  // Analysis Settings
  ANALYSIS: {
    MIN_RELEVANCE: 0.6,
    MIN_CONFIDENCE: 0.7,
    MAX_TOPICS: 5,
    SENTIMENT_SCALE: [-1, 1]
  }
};
```

### File System Constants

```typescript
export const FS_CONSTANTS = {
  // Paths
  PATHS: {
    CACHE: './data/cache',
    LOGS: './logs',
    TEMP: './temp',
    CONFIG: './config'
  },

  // File Settings
  FILES: {
    MAX_SIZE: '1GB',
    COMPRESSION: true,
    ENCODING: 'utf8',
    LINE_ENDING: '\n'
  },

  // Cache Settings
  CACHE: {
    MAX_AGE: '7d',
    CLEANUP_INTERVAL: '1d',
    MIN_AGE: '1h',
    BATCH_SIZE: 100
  },

  // Security
  SECURITY: {
    ALLOWED_PATHS: ['data', 'logs', 'temp'],
    FILE_PERMISSIONS: 0o644,
    DIR_PERMISSIONS: 0o755
  }
};
```

### Error Constants

```typescript
export const ERROR_CONSTANTS = {
  // Error Codes
  CODES: {
    // AI Errors
    AI_TOKEN_LIMIT: 'TOKEN_LIMIT',
    AI_RATE_LIMIT: 'RATE_LIMIT',
    AI_INVALID_RESPONSE: 'INVALID_RESPONSE',
    
    // Security Errors
    ACCESS_DENIED: 'ACCESS_DENIED',
    RATE_LIMITED: 'RATE_LIMITED',
    INVALID_PERMISSION: 'INVALID_PERMISSION',
    
    // Cache Errors
    CACHE_MISS: 'CACHE_MISS',
    CACHE_FULL: 'CACHE_FULL',
    CACHE_EXPIRED: 'CACHE_EXPIRED'
  },

  // Retry Settings
  RETRY: {
    MAX_RETRIES: 3,
    BASE_DELAY: 1000,
    MAX_DELAY: 10000,
    BACKOFF: 2
  },

  // Messages
  MESSAGES: {
    DEFAULT_ERROR: 'An error occurred',
    ACCESS_DENIED: 'You do not have permission',
    RATE_LIMITED: 'Please wait before trying again',
    INVALID_INPUT: 'Invalid input provided'
  }
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
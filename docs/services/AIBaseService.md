[Back to Index](../API.md)

# AIBaseService

The AIBaseService is the foundation class for all AI-related functionality, providing common utilities and model initialization for working with Google's Generative AI (Gemini).

## Overview

AIBaseService establishes the base connection to Google's Generative AI and provides shared functionality for the AI subsystem. It's designed to be extended by specialized AI services rather than used directly.

## Properties

### Shared AI Client Instance

```typescript
protected static genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");
```

A shared instance of the Google Generative AI client, initialized with the API key from environment variables.

### Model Instances

```typescript
protected static defaultModel = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

protected static contextModel = this.genAI.getGenerativeModel({ 
  model: "gemini-2.0-flash",
  generationConfig: {
    temperature: 0.4,
    topP: 0.8,
    topK: 40,
    maxOutputTokens: 2048,
  }
});
```

Pre-configured model instances:
- `defaultModel`: Standard configuration for general-purpose use
- `contextModel`: Optimized configuration for context-aware responses with controlled randomness

## Methods

### `createModel`

```typescript
protected static createModel(config: GenerationConfig): GenerativeModel
```

Creates a model with custom generation configuration.

#### Features:
- Allows custom configuration of model parameters
- Uses the "gemini-2.0-flash" model
- Supports full Generative AI configuration options

#### Parameters:
- `config: GenerationConfig` - Generation configuration parameters

#### Returns:
- `GenerativeModel` - A configured GenerativeModel instance

#### Example:
```typescript
// Within a service that extends AIBaseService
const customModel = this.createModel({
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  maxOutputTokens: 1024,
});
```

### `handleError`

```typescript
protected static handleError(error: any, context: string): string
```

Handles AI generation errors with standardized logging.

#### Features:
- Standardized error handling for AI operations
- Contextual error logging
- Returns user-friendly error message

#### Parameters:
- `error: any` - The error that occurred
- `context: string` - Additional context about where the error occurred

#### Returns:
- `string` - A default error message for the user

#### Example:
```typescript
// Within a service that extends AIBaseService
try {
  // AI operation
} catch (error) {
  return this.handleError(error, 'generating response');
}
```

## Usage Notes

- AIBaseService is designed to be extended rather than used directly
- All specialized AI services inherit from AIBaseService
- Centralized API key management ensures consistent authentication
- The model configurations are optimized for different use cases:
  - Default model: General queries and tasks
  - Context model: Processing conversations with specific context
- Custom models can be created for specific temperature needs
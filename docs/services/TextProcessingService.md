[Back to Index](../API.md)

# TextProcessingService

The TextProcessingService handles text preprocessing, cleaning, and optimization for AI operations.

## Core Features

### Text Processing
- Text normalization
- Stopword removal
- Language detection
- Token optimization

### Content Analysis
- Keyword extraction
- Text segmentation
- Content summarization
- Format standardization

## Method Reference

### normalizeText
```typescript
static normalizeText(
  text: string,
  options?: NormalizationOptions
): string
```

Normalizes text for consistent processing.

#### Parameters:
- `text`: Input text
- `options`: Normalization settings
  - `case`: Case normalization
  - `spaces`: Space normalization
  - `punctuation`: Punctuation handling
  - `unicode`: Unicode normalization

#### Returns:
Normalized text string

### removeStopwords
```typescript
static removeStopwords(
  text: string,
  options?: StopwordOptions
): string
```

Removes common stopwords from text.

#### Parameters:
- `text`: Input text
- `options`: Stopword settings
  - `languages`: Language codes
  - `customStopwords`: Additional stopwords
  - `preserveStructure`: Maintain spacing

#### Returns:
Text with stopwords removed

### extractKeywords
```typescript
static extractKeywords(
  text: string,
  options?: KeywordOptions
): string[]
```

Extracts important keywords from text.

#### Parameters:
- `text`: Source text
- `options`: Extraction options
  - `minLength`: Minimum word length
  - `maxKeywords`: Maximum keywords
  - `weights`: Term importance weights

#### Returns:
Array of extracted keywords

### detectLanguage
```typescript
static detectLanguage(
  text: string,
  options?: LanguageOptions
): LanguageResult
```

Detects the language of text.

#### Parameters:
- `text`: Input text
- `options`: Detection options
  - `confidence`: Minimum confidence
  - `allowedLanguages`: Language whitelist
  - `fallback`: Default language

#### Returns:
Language detection result

## Integration Examples

### Basic Text Processing
```typescript
const normalized = TextProcessingService.normalizeText(text, {
  case: 'lower',
  spaces: true,
  punctuation: 'preserve'
});

const keywords = TextProcessingService.extractKeywords(normalized, {
  minLength: 3,
  maxKeywords: 10
});
```

### Language Processing
```typescript
const lang = TextProcessingService.detectLanguage(text, {
  confidence: 0.8,
  allowedLanguages: ['en', 'es', 'pt']
});

if (lang.confidence > 0.9) {
  const processed = TextProcessingService.removeStopwords(text, {
    languages: [lang.code]
  });
}
```

### Content Optimization
```typescript
// Prepare text for AI processing
const prepared = await TextProcessingService.prepareForAI(text, {
  normalize: true,
  removeStopwords: true,
  maxLength: 1000
});
```

## Error Handling

### Processing Errors
```typescript
try {
  const processed = TextProcessingService.normalizeText(text);
} catch (error) {
  if (error instanceof TextProcessingError) {
    console.error('Processing failed:', error.message);
    return text; // Return original text as fallback
  }
  throw error;
}
```

### Language Detection
```typescript
try {
  const lang = TextProcessingService.detectLanguage(text);
} catch (error) {
  console.warn('Language detection failed:', error);
  return { code: 'en', confidence: 0 }; // Default to English
}
```

## Best Practices

1. **Text Processing**
   - Normalize before analysis
   - Handle unicode properly
   - Preserve important formatting

2. **Performance**
   - Process in batches
   - Cache frequent operations
   - Optimize for length

3. **Language Handling**
   - Support multiple languages
   - Use proper fallbacks
   - Maintain context

## Configuration

```typescript
const TEXT_CONFIG = {
  // Normalization
  CASE_HANDLING: 'lower',
  SPACE_NORMALIZATION: true,
  PUNCTUATION_HANDLING: 'preserve',
  
  // Extraction
  MIN_KEYWORD_LENGTH: 3,
  MAX_KEYWORDS: 20,
  MIN_WORD_FREQUENCY: 2,
  
  // Language
  DEFAULT_LANGUAGE: 'en',
  MIN_CONFIDENCE: 0.7,
  SUPPORTED_LANGUAGES: ['en', 'es', 'pt'],
  
  // Processing
  MAX_TEXT_LENGTH: 10000,
  BATCH_SIZE: 1000,
  CACHE_LIFETIME: 3600
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).
[Back to Index](../API.md)

# TextProcessingService

The TextProcessingService provides text processing operations like keyword extraction, text cleaning, and text chunking. It offers utility functions that help prepare text for AI processing or improve text readability.

## Properties

### Common Words List

```typescript
private static readonly COMMON_WORDS = new Set([
  // Portuguese stopwords
  'e', 'ou', 'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas', 
  'um', 'uma', 'uns', 'umas', 'o', 'a', 'os', 'as', 'para', 'por', 'com', 'sem',
  'que', 'como', 'mais', 'mas', 'já', 'ao', 'esta', 'este', 'esse', 'isso',
  
  // English stopwords
  'the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'from', 'by', 'about',
  'as', 'is', 'it', 'be', 'this', 'that', 'are', 'was', 'were', 'has', 'have'
]);
```

A set of common words (stopwords) in Portuguese and English to exclude from keyword extraction.

## Methods

### `extractKeywords`

```typescript
public static extractKeywords(text: string, minLength: number = 3): string[]
```

Extracts meaningful keywords from a text by removing common words and short terms.

#### Features:
- Removes stopwords in multiple languages
- Filters out terms shorter than minimum length
- Normalizes text to lowercase for consistent matching
- Handles punctuation and whitespace

#### Parameters:
- `text: string` - The input text to extract keywords from
- `minLength: number` - Minimum length for a keyword (default: 3)

#### Returns:
- `string[]` - Array of extracted keywords

#### Example:
```typescript
const keywords = TextProcessingService.extractKeywords(
  "How to implement machine learning algorithms in JavaScript"
);
// Returns: ["implement", "machine", "learning", "algorithms", "javascript"]
```

### `cleanText`

```typescript
public static cleanText(text: string): string
```

Cleans text by removing Markdown formatting characters.

#### Features:
- Removes Markdown characters like *, _, `, etc.
- Preserves the actual content
- Trims leading/trailing whitespace
- Results in plain text suitable for AI processing

#### Parameters:
- `text: string` - The text to clean

#### Returns:
- `string` - Cleaned text without markdown characters

#### Example:
```typescript
const cleanedText = TextProcessingService.cleanText(
  "This is **bold** and _italicized_ text with `code` formatting"
);
// Returns: "This is bold and italicized text with code formatting"
```

### `truncateText`

```typescript
public static truncateText(text: string, maxLength: number): string
```

Truncates text to specified maximum length, preserving word boundaries.

#### Features:
- Ensures truncated text ends at word boundaries
- Adds ellipsis to indicate truncation
- Handles edge cases for very short maxLength values
- Returns original text if already within length limits

#### Parameters:
- `text: string` - The input text to truncate
- `maxLength: number` - Maximum length for the output text

#### Returns:
- `string` - Truncated text ending at a word boundary

#### Example:
```typescript
const truncated = TextProcessingService.truncateText(
  "This is a long sentence that needs to be truncated properly.", 
  20
);
// Returns: "This is a long..."
```

### `splitTextIntoChunks`

```typescript
public static splitTextIntoChunks(text: string, maxChunkSize: number): string[]
```

Splits text into chunks of specified maximum size while preserving paragraph breaks.

#### Features:
- Prioritizes splitting at paragraph boundaries
- Falls back to sentence boundaries if needed
- Uses word boundaries as last resort
- Preserves text integrity and readability
- Ensures chunks are within size limits

#### Parameters:
- `text: string` - The text to split into chunks
- `maxChunkSize: number` - Maximum size for each chunk

#### Returns:
- `string[]` - Array of text chunks

#### Example:
```typescript
const chunks = TextProcessingService.splitTextIntoChunks(
  longText,
  1000
);
console.log(`Split into ${chunks.length} readable chunks`);
```

## Usage Recommendations

- Use `extractKeywords` for topic extraction and relevance matching
- Apply `cleanText` before sending text to AI processing to remove formatting
- Use `truncateText` for UI display where space is limited
- Apply `splitTextIntoChunks` when processing long texts that exceed API limits
- Consider expanding the COMMON_WORDS set for your specific domain needs
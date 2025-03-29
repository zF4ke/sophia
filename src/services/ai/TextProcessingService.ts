import { AIBaseService } from "./AIBaseService";

/**
 * Service for text processing operations like keyword extraction and text cleaning
 */
export class TextProcessingService extends AIBaseService {
  /**
   * Set of common words to exclude from keyword extraction in multiple languages
   * @private
   */
  private static readonly COMMON_WORDS = new Set([
    // Portuguese stopwords
    'e', 'ou', 'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas', 
    'um', 'uma', 'uns', 'umas', 'o', 'a', 'os', 'as', 'para', 'por', 'com', 'sem',
    'que', 'como', 'mais', 'mas', 'já', 'ao', 'esta', 'este', 'esse', 'isso',
    
    // English stopwords
    'the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'from', 'by', 'about',
    'as', 'is', 'it', 'be', 'this', 'that', 'are', 'was', 'were', 'has', 'have'
  ]);

  /**
   * Extracts meaningful keywords from a text by removing common words and short terms
   * @param text - The input text to extract keywords from
   * @param minLength - Minimum length for a keyword (default: 3)
   * @returns Array of extracted keywords
   */
  public static extractKeywords(text: string, minLength: number = 3): string[] {
    return text
      .toLowerCase()
      .split(/[\s,.-]+/)
      .filter(word => word.length >= minLength && !this.COMMON_WORDS.has(word));
  }
  
  /**
   * Cleans text by removing Markdown formatting characters
   * @param text - The text to clean
   * @returns Cleaned text without markdown characters
   */
  public static cleanText(text: string): string {
    return text.replace(/[\\`*_{}[\]()#+\-.!]/g, '').trim();
  }

  /**
   * Truncates text to specified maximum length, preserving word boundaries
   * @param text - The input text to truncate
   * @param maxLength - Maximum length for the output text
   * @returns Truncated text ending at a word boundary
   */
  public static truncateText(text: string, maxLength: number): string {
    if (text.length <= maxLength) return text;
    
    // Find a good breaking point at word boundary
    const breakPoint = text.lastIndexOf(' ', maxLength);
    return breakPoint > 0 ? text.substring(0, breakPoint) + '...' : text.substring(0, maxLength) + '...';
  }
  
  /**
   * Splits text into chunks of specified maximum size while preserving paragraph breaks
   * @param text - The text to split into chunks
   * @param maxChunkSize - Maximum size for each chunk
   * @returns Array of text chunks
   */
  public static splitTextIntoChunks(text: string, maxChunkSize: number): string[] {
    const chunks: string[] = [];
    
    if (text.length <= maxChunkSize) {
      return [text];
    }
    
    let currentIndex = 0;
    
    while (currentIndex < text.length) {
      // Try to split at paragraph or sentence boundaries if possible
      let endIndex = currentIndex + maxChunkSize;
      
      if (endIndex < text.length) {
        // Try paragraph break first
        const paragraphBreak = text.lastIndexOf('\n\n', endIndex);
        if (paragraphBreak > currentIndex && paragraphBreak - currentIndex >= maxChunkSize / 2) {
          endIndex = paragraphBreak;
        } else {
          // Then try sentence break
          const sentenceBreak = Math.max(
            text.lastIndexOf('. ', endIndex),
            text.lastIndexOf('! ', endIndex),
            text.lastIndexOf('? ', endIndex)
          );
          
          if (sentenceBreak > currentIndex && sentenceBreak - currentIndex >= maxChunkSize / 3) {
            endIndex = sentenceBreak + 1; // Include the punctuation
          } else {
            // Fall back to word boundary
            const wordBreak = text.lastIndexOf(' ', endIndex);
            if (wordBreak > currentIndex) {
              endIndex = wordBreak;
            }
          }
        }
      }
      
      chunks.push(text.slice(currentIndex, endIndex).trim());
      currentIndex = endIndex;
    }
    
    return chunks;
  }
}
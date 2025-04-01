import { DISCORD } from "@/utils/constants";
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
   * Splits a long message into chunks that fit within Discord's message limit
   * @param message The message to split
   * @param limit The maximum length per chunk (default: Discord's message limit)
   * @returns Array of message chunks
   */
  public static splitLongMessage(message: string, limit: number = DISCORD.MESSAGE_LIMIT): string[] {
    const chunks: string[] = [];
    
    // If message is already within limit, return it as is
    if (message.length <= limit) {
        return [message];
    }
    
    let currentChunk = '';
    // Split by paragraphs (double newlines) first to maintain logical structure
    const paragraphs = message.split('\n\n');
    
    for (const paragraph of paragraphs) {
        // If adding this paragraph would exceed the limit, push current chunk and start a new one
        if ((currentChunk + paragraph + '\n\n').length > limit) {
            // If the paragraph itself is too long, split it further
            if (paragraph.length > limit) {
                // First push current chunk if it exists
                if (currentChunk) {
                    chunks.push(currentChunk);
                    currentChunk = '';
                }
                
                // Split long paragraph by sentences and try to keep sentences together
                const sentences = paragraph.split(/(?<=\.|\?|\!) /);
                for (const sentence of sentences) {
                    if ((currentChunk + sentence + ' ').length <= limit) {
                        currentChunk += sentence + ' ';
                    } else {
                        // If the sentence itself is too long, split by words
                        if (sentence.length > limit) {
                            if (currentChunk) {
                                chunks.push(currentChunk);
                                currentChunk = '';
                            }
                            
                            // Split by words
                            let words = sentence.split(' ');
                            for (const word of words) {
                                if ((currentChunk + word + ' ').length <= limit) {
                                    currentChunk += word + ' ';
                                } else {
                                    chunks.push(currentChunk);
                                    currentChunk = word + ' ';
                                }
                            }
                        } else {
                            chunks.push(currentChunk);
                            currentChunk = sentence + ' ';
                        }
                    }
                }
            } else {
                // Paragraph fits in a new chunk
                chunks.push(currentChunk);
                currentChunk = paragraph + '\n\n';
            }
        } else {
            // Add paragraph to current chunk
            currentChunk += paragraph + '\n\n';
        }
    }
    
    // Add the last chunk if it's not empty
    if (currentChunk) {
        chunks.push(currentChunk);
    }
    
    return chunks;
  }
}
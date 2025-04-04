import { Message } from "discord.js";
import { AIBaseService } from "./AIBaseService";
import { TextProcessingService } from "./TextProcessingService";

/**
 * Service for managing conversation contexts for AI interactions
 */
export class ContextManagementService extends AIBaseService {
  /**
   * Default maximum context length in characters
   * @private
   */
  private static readonly DEFAULT_MAX_CONTEXT_CHARS = 50000;

  /**
   * Selects the most relevant conversations for providing context to AI
   * @param conversations - Array of conversations to select from
   * @param prompt - User prompt to base relevance on
   * @param maxChars - Maximum character limit for context (default: 50000)
   * @returns Array of selected conversations within character limit
   */
  public static selectConversationsForContext(
    conversations: Message[][], 
    prompt: string, 
    maxChars: number = this.DEFAULT_MAX_CONTEXT_CHARS
  ): Message[][] {
    // Extract keywords from prompt for matching
    const promptKeywords = TextProcessingService.extractKeywords(prompt);
    
    // Score conversations based on relevance to prompt
    const scoredConversations = conversations.map(conversation => {
      const combinedText = conversation.map(msg => msg.content).join(' ').toLowerCase();
      
      // Determine keyword match score
      const keywordScore = promptKeywords.reduce((score, keyword) => {
        return score + (combinedText.includes(keyword.toLowerCase()) ? 1 : 0);
      }, 0);
      
      // Consider conversation length/complexity
      const lengthScore = Math.min(5, conversation.length / 2);
      
      // Calculate total character length
      const charLength = conversation.reduce((sum, msg) => 
        sum + msg.author.username.length + 2 + msg.content.length, 0);
      
      return {
        conversation,
        score: keywordScore * 2 + lengthScore, // Weight keywords higher
        charLength
      };
    });
    
    // Sort by relevance score (descending)
    scoredConversations.sort((a, b) => b.score - a.score);
    
    // Select conversations within character limit
    const selected: Message[][] = [];
    let totalChars = 0;
    
    for (const item of scoredConversations) {
      if (totalChars + item.charLength <= maxChars) {
        selected.push(item.conversation);
        totalChars += item.charLength;
      } else {
        // Always include at least one conversation even if it exceeds limit
        if (selected.length === 0) {
          selected.push(item.conversation);
        }
        break;
      }
    }
    
    return selected;
  }

  /**
   * Formats conversations into a structured context string for AI
   * @param conversations - Array of conversations to format
   * @returns Formatted context string
   */
  public static formatConversationsAsContext(conversations: Message[][]): string {
    let context = conversations.map((conversation, index) => {
      const formattedConversation = conversation.map(msg => 
        `${msg.author.username}: ${msg.content.trim()}`
      ).join('\n');
      
      return `[Conversa ${index + 1}]\n${formattedConversation}`;
    }).join('\n\n');
    // add separator at the end
    context += '\n\n---------------\n\n';

    return context;
  }

  /**
   * Formats messages into a structured context string for AI
   * @param messages - Array of messages to format
   * @param includeBots - Flag to include bot messages in context
   * @returns Formatted context string
   */
  public static formatMessagesAsContext(
    messages: Message[], 
    includeBots: boolean = false,
    reverse: boolean = false
  ): string {
    const filteredMessages = messages.filter(msg => 
      includeBots || !msg.author.bot
    );
    
    // return filteredMessages.map(msg => 
    //   `${msg.author.username}: ${msg.content.trim()}`
    // ).join('\n');
    messages = reverse ? filteredMessages.reverse() : filteredMessages;

    // if a message has more than one line, add """ around it """
    const formattedMessages = messages.map(msg => {
      const content = msg.content.trim();
      return content.includes('\n') ? 
        `Mensagem enviada por ${msg.author.username}: """\n${content}\n"""` : 
        `Mensagem enviada por ${msg.author.username}: ${content}`;
    }).join('\n');

    // add separator at the end
    return `${formattedMessages}\n\n---------------\n\n`;
  }
  
  /**
   * Optimizes context by focusing on most relevant parts if context exceeds token limits
   * @param conversations - Array of conversations
   * @param prompt - User prompt
   * @param maxTokens - Approximate maximum token count for context
   * @returns Optimized context string
   */
  public static optimizeContextForTokenLimit(
    conversations: Message[][], 
    prompt: string, 
    maxTokens: number = 8000
  ): string {
    // Approximate token count (rough estimate: ~4 chars per token)
    const estimateTokens = (text: string): number => Math.ceil(text.length / 4);
    
    // Sort conversations by relevance to prompt
    const selected = this.selectConversationsForContext(conversations, prompt);
    
    // Format and check if within token limit
    let formattedContext = this.formatConversationsAsContext(selected);
    let estimatedTokens = estimateTokens(formattedContext);
    
    // If context is too large, trim conversations while preserving most relevant parts
    if (estimatedTokens > maxTokens) {
      // Extract most relevant messages from each conversation
      const trimmedConversations = selected.map(conversation => {
        // Prioritize messages with keyword matches to the prompt
        const promptKeywords = TextProcessingService.extractKeywords(prompt);
        
        // Score messages by relevance
        const scoredMessages = conversation.map(msg => {
          const keywordMatches = promptKeywords.filter(kw => 
            msg.content.toLowerCase().includes(kw)
          ).length;
          
          const contentLength = msg.content.split(/\s+/).length;
          
          return {
            message: msg,
            score: keywordMatches * 3 + Math.min(2, contentLength / 10)
          };
        });
        
        // Sort by relevance score
        scoredMessages.sort((a, b) => b.score - a.score);
        
        // Take top 50% of messages or at least 3 messages, whichever is greater
        const numToKeep = Math.max(3, Math.floor(conversation.length * 0.5));
        return scoredMessages.slice(0, numToKeep).map(item => item.message);
      });
      
      // Reformat with reduced context
      formattedContext = this.formatConversationsAsContext(trimmedConversations);
      estimatedTokens = estimateTokens(formattedContext);
      
      // If still too large, take fewer conversations
      if (estimatedTokens > maxTokens) {
        const reducedSelectedConversations = selected.slice(
          0, 
          Math.max(1, Math.floor(selected.length * 0.6))
        );
        formattedContext = this.formatConversationsAsContext(reducedSelectedConversations);
      }
    }
    
    return formattedContext;
  }
}
import { Message, ChatInputCommandInteraction } from "discord.js";
import { ConversationWithContext } from "../types/conversation";
import { AIBaseService } from "./ai/AIBaseService";
import { ConversationAnalysisService } from "./ai/ConversationAnalysisService";
import { TextProcessingService } from "./ai/TextProcessingService";
import { ContextManagementService } from "./ai/ContextManagementService";
import { ResponseGenerationService } from "./ai/ResponseGenerationService";

/**
 * Main service for AI-related operations that coordinates specialized AI services
 */
export class AIService extends AIBaseService {
  /**
   * Analyzes conversations for relevance to a specific topic
   * @param conversations - Array of message arrays representing conversations
   * @param topic - The topic to analyze relevance against
   * @param interaction - Discord interaction for updating status
   * @returns Array of relevant conversations with context and relevance scores
   */
  public static async analyzeConversations(
    conversations: Message[][],
    topic: string,
    interaction: ChatInputCommandInteraction
  ): Promise<ConversationWithContext[]> {
    return ConversationAnalysisService.analyzeConversations(
      conversations, 
      topic, 
      interaction
    );
  }

  /**
   * Performs keyword-based search across conversations when AI search fails
   * @param conversations - Array of conversations to search through
   * @param topic - Topic to search for
   * @returns Array of relevant conversations with context and scores
   */
  public static fallbackKeywordSearch(
    conversations: Message[][], 
    topic: string
  ): ConversationWithContext[] {
    return ConversationAnalysisService.fallbackKeywordSearch(conversations, topic);
  }

  /**
   * Extracts meaningful keywords from text
   * @param text - The input text to extract keywords from
   * @returns Array of extracted keywords
   */
  public static extractKeywords(text: string): string[] {
    return TextProcessingService.extractKeywords(text);
  }

  /**
   * Selects relevant conversations to use as context for an AI prompt
   * @param conversations - Array of conversations to select from
   * @param prompt - The user prompt to match against
   * @param maxChars - Maximum character limit for context
   * @returns Selected conversations within character limit
   */
  public static selectConversationsForContext(
    conversations: Message[][], 
    prompt: string, 
    maxChars: number = 50000
  ): Message[][] {
    return ContextManagementService.selectConversationsForContext(
      conversations, 
      prompt, 
      maxChars
    );
  }

  /**
   * Formats conversations into a structured context string
   * @param conversations - Array of conversations to format
   * @returns Formatted context string
   */
  public static formatConversationsAsContext(conversations: Message[][]): string {
    return ContextManagementService.formatConversationsAsContext(conversations);
  }

  /**
   * Formats messages into a structured context string
   * @param messages - Array of messages to format
   * @param includeBots - Flag to include bot messages in context
   * @returns Formatted context string
   */
  public static formatMessagesAsContext(
    messages: Message[], 
    includeBots: boolean = false,
    reverse: boolean = false
  ): string {
    return ContextManagementService.formatMessagesAsContext(messages, includeBots, reverse);
  }

  /**
   * Optimizes context for token limitations while preserving relevance
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
    return ContextManagementService.optimizeContextForTokenLimit(
      conversations, 
      prompt, 
      maxTokens
    );
  }

  /**
   * Cleans text by removing Markdown formatting characters
   * @param text - The text to clean
   * @returns Cleaned text without markdown characters
   */
  public static cleanText(text: string): string {
    return TextProcessingService.cleanText(text);
  }

  /**
   * Generates a custom response using a template and parameters
   * @param promptTemplate - Template string with placeholders
   * @param params - Object containing values to replace placeholders
   * @param temperature - Temperature parameter for response randomness (0.0-1.0)
   * @returns Promise with the generated response
   */
  public static async generateCustomResponse(
    promptTemplate: string, 
    params: Record<string, string>,
    temperature: number = 0.4
  ): Promise<string> {
    return ResponseGenerationService.generateCustomResponse(
      promptTemplate, 
      params, 
      temperature
    );
  }

  /**
   * Generates a response to a user prompt using conversation context
   * @param prompt - User query or instruction
   * @param context - Conversation context to inform the response
   * @param additionalInstructions - Additional instructions for the AI
   * @returns Promise with the generated response
   */
  public static async generateConversationResponse(
    question: string, 
    context: string,
    additionalInstructions: string = "",
  ): Promise<string> {
    return ResponseGenerationService.generateConversationResponse(question, context, additionalInstructions);
  }

  /**
   * Generates a contextual response to a user prompt using conversation context
   * @param prompt - User query or instruction
   * @param context - Conversation context to inform the response
   * @param options - Options for response generation
   * @returns Promise with the generated response
   */
  public static async generateContextualResponse(
    question: string, 
    context: string,
    options: {
      additionalInstructions?: string,
      extremelyLongAnswer?: boolean,
    } = {},
  ): Promise<string> {
    return ResponseGenerationService.generateContextualResponse(question, context, options);
  }

  /**
   * Generates a comprehensive response to a question using web search capabilities
   * and optionally including chat context
   * @param question - The question to answer
   * @param context - Optional chat context to supplement web information
   * @param additionalInstructions - Additional instructions for the AI
   * @returns Promise with the generated response
   */
  public static async generateWebSearchResponse(
    question: string,
    context: string = "",
    additionalInstructions: string = "",
  ): Promise<string> {
    return ResponseGenerationService.generateWebSearchResponse(question, context, additionalInstructions);
  }
}
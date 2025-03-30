import { AIBaseService } from "./AIBaseService";
import { ContextManagementService } from "./ContextManagementService";

/**
 * Service for generating AI responses based on user prompts and contexts
 */
export class ResponseGenerationService extends AIBaseService {
  /**
   * Generates a contextual response to a user prompt using provided conversation context
   * @param prompt - User query or instruction
   * @param context - Conversation context to inform the response
   * @param additionalInstructions - Optional additional instructions for the AI
   * @param maxLength - Maximum target length for response (default: 1000 characters)
   * @returns Promise with the generated response
   */
  public static async generateContextualResponse(
    prompt: string, 
    context: string,
    additionalInstructions: string = "",
    maxLength: number = 1000
  ): Promise<string> {
    try {
      // Add conciseness instruction
      const conciseInstruction = `Seja conciso e direto. Limite sua resposta a aproximadamente ${maxLength} caracteres.`;
      const combinedInstructions = additionalInstructions 
        ? `${additionalInstructions}\n${conciseInstruction}` 
        : conciseInstruction;
      
      // Create structured prompt with context
      const aiPrompt = ContextManagementService.createContextualPrompt(
        prompt, 
        context,
        combinedInstructions
      );
      
      // Generate AI response
      const result = await this.contextModel.generateContent(aiPrompt);
      const response = result.response.text();
      
      return response;
    } catch (error) {
      return this.handleError(
        error, 
        'generating contextual response'
      );
    }
  }
  
  /**
   * Generates a response using a custom prompt template and specified parameters
   * @param promptTemplate - Template string with placeholders
   * @param params - Object containing values to replace placeholders
   * @param temperature - Temperature parameter for controlling response randomness (0.0-1.0)
   * @returns Promise with the generated response
   */
  public static async generateCustomResponse(
    promptTemplate: string, 
    params: Record<string, string>,
    temperature: number = 0.4
  ): Promise<string> {
    try {
      // Replace placeholders in template with parameter values
      let finalPrompt = promptTemplate;
      for (const [key, value] of Object.entries(params)) {
        finalPrompt = finalPrompt.replace(new RegExp(`\\{\\{${key}\\}\\}`, 'g'), value);
      }
      
      // Create custom model with specified temperature
      const customModel = this.createModel({
        temperature,
        topP: 0.8,
        topK: 40,
        maxOutputTokens: 2048,
      });
      
      // Generate AI response
      const result = await customModel.generateContent(finalPrompt);
      return result.response.text();
    } catch (error) {
      return this.handleError(
        error,
        'generating custom response'
      );
    }
  }
  
  /**
   * Summarizes a long text or conversation
   * @param text - Text to be summarized
   * @param maxLength - Target maximum length for summary
   * @param focusTopics - Optional array of topics to focus on
   * @returns Promise with the generated summary
   */
  public static async generateSummary(
    text: string, 
    maxLength: number = 500,
    focusTopics: string[] = []
  ): Promise<string> {
    try {
      let prompt = `
      Resumir o seguinte texto de forma concisa e clara. 
      O resumo deve ter aproximadamente ${maxLength} caracteres ou menos.
      
      ${focusTopics.length > 0 
        ? `Foque nos seguintes aspectos: ${focusTopics.join(', ')}.` 
        : 'Capture os pontos e informações mais importantes.'}
      
      Texto para resumir:
      ${text}
      `;
      
      const result = await this.contextModel.generateContent(prompt);
      return result.response.text();
    } catch (error) {
      return this.handleError(
        error,
        'generating summary'
      );
    }
  }
  
  /**
   * Generates a follow-up response based on previous conversation and new input
   * @param previousMessages - Array of previous message pairs [user, assistant]
   * @param newUserInput - Latest user input to respond to
   * @returns Promise with the follow-up response
   */
  public static async generateConversationResponse(
    previousMessages: {role: string, content: string}[],
    newUserInput: string
  ): Promise<string> {
    try {
      // Format conversation history for context
      const conversationHistory = previousMessages.map(msg => 
        `${msg.role}: ${msg.content}`
      ).join('\n\n');
      
      const prompt = `
      Esta é uma conversa em andamento. Responda à última mensagem do usuário considerando o contexto da conversa.
      
      Histórico da conversa:
      ${conversationHistory}
      
      Nova mensagem do usuário:
      ${newUserInput}
      
      Responda de forma natural, concisa e adequada ao contexto da conversa.
      `;
      
      const result = await this.contextModel.generateContent(prompt);
      return result.response.text();
    } catch (error) {
      return this.handleError(
        error,
        'generating conversation response'
      );
    }
  }
  
  /**
   * Generates a response based on web search results and optional context
   * @param prompt - Combined prompt with question and optional context
   * @param temperature - Temperature parameter for response randomness (0.0-1.0)
   * @returns Promise with the generated response from web search and/or AI reasoning
   */
  public static async generateWebSearchResponse(
    prompt: string,
    temperature: number = 0.8
  ): Promise<string> {
    try {
      // Create a model instance with higher temperature for diverse and opinionated responses
      const webSearchModel = this.createModel({
        temperature,
        topP: 0.9,
        topK: 40,
        //maxOutputTokens: 4096, // Larger output for comprehensive answers
        maxOutputTokens: 2048, // Adjusted for performance
      });
      
      // Generate AI response with web search capability and opinion generation
      const result = await webSearchModel.generateContent(prompt);
      return result.response.text();
    } catch (error) {
      return this.handleError(
        error,
        'generating comprehensive response'
      );
    }
  }
}
import { AIBaseService } from "./AIBaseService";
import { ContextManagementService } from "./ContextManagementService";

/**
 * Service for generating AI responses based on user prompts and contexts
 */
export class ResponseGenerationService extends AIBaseService {
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
      Esta é uma conversa em entre amigos. Responda à última mensagem do usuário considerando o contexto da conversa.

      Histórico da conversa:
      ${conversationHistory}

      Nova mensagem do usuário:
      ${newUserInput}

      Responda de forma natural, casual e adequada ao contexto da conversa. Não faça respostas formais ou técnicas. Não faça respostas muito longas, pois fica chato de ler.
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
   * Generates a contextual response to a user prompt using provided conversation context
   * @param prompt - User query or instruction
   * @param context - Conversation context to inform the response
   * @param additionalInstructions - Optional additional instructions for the AI
   * @param maxLength - Maximum target length for response (default: 1000 characters)
   * @returns Promise with the generated response
   */
  public static async generateContextualResponse(
    question: string, 
    context: string,
    additionalInstructions: string = "",
  ): Promise<string> {
    try {
      const prompt = `
            Você tem acesso a conversas de um canal do Discord. Use essas conversas como contexto para responder à pergunta ou executar a instrução do usuário.
            
            Contexto das conversas:
            ${context}
            
            ${additionalInstructions ? additionalInstructions + "\n\n" : ""}
            Pergunta/instrução do usuário:
            ${question}
            
            Diretrizes:
            - Base sua resposta no contexto fornecido
            - Se o contexto não contiver informações relevantes, diga isso claramente
            - Cite partes específicas do contexto para justificar sua resposta quando relevante
            - Seja conciso mas completo
            - Formate sua resposta de forma clara e organizada
            - Não inclua prefixos como "Baseado no contexto" ou "Resposta:"
        `;
      
      // Generate AI response
      const result = await this.contextModel.generateContent(prompt);
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
   * Generates a response based on web search results and optional context
   * @param prompt - Combined prompt with question and optional context
   * @param temperature - Temperature parameter for response randomness (0.0-1.0)
   * @returns Promise with the generated response from web search and/or AI reasoning
   */
  public static async generateWebSearchResponse(
    question: string,
    context: string = "",
    additionalInstructions: string = "",
  ): Promise<string> {
    try {
      const result = await this.contextModel.generateContent(question);
      return result.response.text();
    } catch (error) {
      return this.handleError(
        error,
        'generating comprehensive response'
      );
    }
  }
}
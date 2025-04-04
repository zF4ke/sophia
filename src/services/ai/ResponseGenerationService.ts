import { AIBaseService } from "./AIBaseService";
import { ContextManagementService } from "./ContextManagementService";
import { PersonalityService } from "./PersonalityService";

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
      
      // Add personality to the prompt
      const style = PersonalityService.determineResponseStyle(finalPrompt);
      finalPrompt = PersonalityService.adjustPromptForPersonality(finalPrompt, style);
      
      // Create custom model with specified temperature
      const customModel = this.createModel({
        temperature,
        topP: 0.8,
        topK: 40,
        maxOutputTokens: 2048,
      });
      
      // Generate AI response
      const result = await customModel.generateContent(finalPrompt);
      const response = result.response.text();
      
      return response;
    } catch (error) {
      return this.handleError(
        error,
        'generating custom response'
      );
    }
  }
  
  /**
   * Generates a follow-up response based on previous conversation and new input
   * @param question - User's question or instruction
   * @param context - Conversation context to inform the response
   * @param additionalInstructions - Optional additional instructions for the AI
   * @returns Promise with the generated response
   */
  public static async generateConversationResponse(
    question: string,
    context: string,
    additionalInstructions: string = "",
  ): Promise<string> {
    try {
      // Determine appropriate style based on conversation history and new input
      const style = PersonalityService.determineResponseStyle(question, context);

      const basePrompt = `
            Você tem acesso a conversas de um canal do Discord.
            Responda à última mensagem do usuário considerando o contexto da conversa, para continuar a conversa.
            
      Histórico da conversa:
            ${context}
            
            ${additionalInstructions ? additionalInstructions + "\n\n" : ""}
            Nova mensagem do usuário:
            ${question}
            
            Diretrizes:
            - Responda como uma conversa entre colegas
            - É provável que a conversa seja informal e amigável
            - Continue a conversa na mesma linha de raciocínio, mantendo o tom e o estilo
            - Não inclua prefixos como "Baseado no contexto" ou "Resposta:"
            - Evite repetir o que já foi dito anteriormente
            - Se o usuário mudar de assunto, adapte sua resposta para o novo tópico
            - Avoid repeating the same phrases or sentences
            - Don't always end with a question
            - Resposta mais curta é melhor
      `;

      const finalPrompt = PersonalityService.adjustPromptForPersonality(basePrompt, style);
      
      const result = await this.contextModel.generateContent(finalPrompt);
      const response = result.response.text();
      
      return response;
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
      const style = PersonalityService.determineResponseStyle(question, context);

      const basePrompt = `
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
            - Seja concisa mas completa
            - Formate sua resposta de forma clara e organizada
            - Não inclua prefixos como "Baseado no contexto" ou "Resposta:" ou "Sophia: ", etc. Apenas responda diretamente à pergunta ou instrução do usuário
      `;

      const finalPrompt = PersonalityService.adjustPromptForPersonality(basePrompt, style);
      
      // Generate AI response
      const result = await this.contextModel.generateContent(finalPrompt);
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
   * @param question - User's question or instruction
   * @param context - Optional context to inform the response
   * @param additionalInstructions - Optional additional instructions for the AI
   * @returns Promise with the generated response from web search and/or AI reasoning
   */
  public static async generateWebSearchResponse(
    question: string,
    context: string = "",
    additionalInstructions: string = "",
  ): Promise<string> {
    try {
      const style = PersonalityService.determineResponseStyle(question, context);

      const basePrompt = `
        Responda à pergunta do usuário usando seu conhecimento, conhecimento da web, e o contexto fornecido (se houver).
        
        ${context ? `Contexto disponível:\n${context}\n\n` : ''}
        ${additionalInstructions ? `${additionalInstructions}\n\n` : ''}
        
        Pergunta do usuário:
        ${question}
      `;

      const finalPrompt = PersonalityService.adjustPromptForPersonality(basePrompt, style);

      const result = await this.contextModel.generateContent(finalPrompt);
      const response = result.response.text();
      
      return response;
    } catch (error) {
      return this.handleError(
        error,
        'generating web search response'
      );
    }
  }
}
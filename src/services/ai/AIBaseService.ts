import { GoogleGenerativeAI, GenerativeModel, GenerationConfig } from "@google/generative-ai";

/**
 * Base service for AI functionality providing model initialization and common utilities
 */
export class AIBaseService {
  /**
   * Google Generative AI client instance
   * @protected
   */
  protected static genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");
  
  /**
   * Default Gemini model for general use
   * @protected
   */
  protected static defaultModel = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
  
  /**
   * Model configured for context-aware responses with specific generation parameters
   * @protected
   */
  protected static contextModel = this.genAI.getGenerativeModel({ 
    model: "gemini-2.0-flash",
    generationConfig: {
      temperature: 0.4,
      topP: 0.8,
      topK: 40,
      //maxOutputTokens: 2048,
      maxOutputTokens: 1024,
    }
  });

  /**
   * Model configured for long responses with specific generation parameters
   * @protected
   */
  protected static longResponseModel = this.genAI.getGenerativeModel({
    model: "gemini-2.0-flash",
    generationConfig: {
      temperature: 0.4,
      topP: 0.8,
      topK: 40,
      maxOutputTokens: 8192,
    }
  });
  
  /**
   * Creates a model with custom generation configuration
   * @param config - Generation configuration parameters
   * @returns A configured GenerativeModel instance
   */
  protected static createModel(config: GenerationConfig): GenerativeModel {
    return this.genAI.getGenerativeModel({
      model: "gemini-2.0-flash",
      generationConfig: config
    });
  }
  
  /**
   * Handles AI generation errors with standardized logging
   * @param error - The error that occurred
   * @param context - Additional context about where the error occurred
   * @returns A default error message for the user
   */
  protected static handleError(error: any, context: string): string {
    console.error(`AI Error in ${context}:`, error);
    return "Desculpe, ocorreu um erro ao processar sua solicitação. Tente novamente mais tarde.";
  }
}
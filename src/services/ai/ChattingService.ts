import { ChatInputCommandInteraction, TextChannel } from "discord.js";
import { MessageService } from "../MessageService";
import { AIService } from "../AIService";
import { TextProcessingService } from "./TextProcessingService";

export class ChattingService {
    /**
     * Generates an AI response using channel messages as context
     * @param channel - The Discord text channel to use as context
     * @param prompt - The user's message/prompt
     * @param options - Configuration options for the conversation
     * @returns The AI's response
     */
    public static async generateChatResponse(
        channel: TextChannel,
        prompt: string,
        options: {
            limit?: number;
            includeBots?: boolean;
            userName?: string;
            interaction?: ChatInputCommandInteraction;
            additionalContext?: string;
        } = {}
    ): Promise<string> {
        const {
            limit = 100,
            includeBots = true,
            userName,
            interaction
        } = options;

        const messages = await MessageService.fetchMessages(channel, limit);
        const contextText = AIService.formatMessagesAsContext(messages, includeBots) + "\n" + (options.additionalContext || "");
            
        const promptContentWithoutMention = prompt.replace(/<@!?[0-9]+>/, "").trim();
        const convertedPrompt = await TextProcessingService.convertMentionsToNames(promptContentWithoutMention, interaction?.user.client!);
        
        const processedPrompt = userName ? 
            TextProcessingService.addAuthorToQuestion(convertedPrompt, userName) : 
            convertedPrompt

        const response = await AIService.generateConversationResponse(processedPrompt, contextText);
        const finalResponse = TextProcessingService.removeTrailingQuotes(response).trim();

        return finalResponse;
    }
}

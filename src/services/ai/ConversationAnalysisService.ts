import type { ChatInputCommandInteraction, Message } from "discord.js";
import type { ConversationWithContext } from "@/types/conversation";
import { TextProcessingService } from "@/services/ai/TextProcessingService";

export class ConversationAnalysisService {
    public static async analyzeConversations(
        conversations: Message[][],
        topic: string,
        _interaction: ChatInputCommandInteraction
    ): Promise<ConversationWithContext[]> {
        return this.fallbackKeywordSearch(conversations, topic);
    }

    public static fallbackKeywordSearch(
        conversations: Message[][],
        topic: string
    ): ConversationWithContext[] {
        const keywords = TextProcessingService.extractKeywords(topic);
        return conversations
            .map((messages) => {
                const haystack = messages.map((message) => message.content.toLowerCase()).join(" ");
                const score = keywords.reduce(
                    (sum, keyword) => sum + (haystack.includes(keyword) ? 1 : 0),
                    0
                );
                return {
                    messages,
                    relevanceScore: Math.min(10, score * 2),
                };
            })
            .filter((conversation) => conversation.relevanceScore > 0)
            .sort((a, b) => b.relevanceScore - a.relevanceScore);
    }
}

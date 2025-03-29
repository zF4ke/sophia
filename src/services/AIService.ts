import { GoogleGenerativeAI } from "@google/generative-ai";
import { Message, ChatInputCommandInteraction } from "discord.js";
import { ConversationWithContext, AIAnalysisResult } from "../types/conversation";
import { EMOJIS } from "../utils/constants";

export class AIService {
    private static genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");
    private static model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    private static calculateBatchSize(totalConversations: number): number {
        if (totalConversations <= 10) return 10;
        if (totalConversations <= 50) return 25;
        if (totalConversations <= 100) return 50;
        if (totalConversations <= 500) return 100;
        return 200;
    }

    public static async analyzeConversations(
        conversations: Message[][],
        topic: string,
        interaction: ChatInputCommandInteraction
    ): Promise<ConversationWithContext[]> {
        const relevantConversations: ConversationWithContext[] = [];
        const batchSize = this.calculateBatchSize(conversations.length);

        for (let i = 0; i < conversations.length; i += batchSize) {
            const batch = conversations.slice(i, i + batchSize);
            const batchResults = await this.analyzeBatch(batch, topic);

            // Update progress every other batch
            if (i % (batchSize * 2) === 0) {
                await interaction.editReply(
                    `${EMOJIS.search} Analisando conversas... (${i}/${conversations.length} processadas)`
                );
            }

            batch.forEach((conversation, index) => {
                const result = batchResults[index];
                if (result?.isRelevant) {
                    relevantConversations.push({
                        messages: conversation,
                        relevanceScore: result.relevanceScore
                    });
                }
            });
        }

        return relevantConversations.sort((a, b) => b.relevanceScore - a.relevanceScore);
    }

    private static async analyzeBatch(conversations: Message[][], topic: string): Promise<AIAnalysisResult[]> {
        try {
            const batchText = conversations.map((conversation, index) => {
                const conversationText = conversation
                    .map(msg => `${msg.author.username}: ${msg.content}`)
                    .join("\n");
                return `[Conversation ${index + 1}]\n${conversationText}\n`;
            }).join("\n---\n");

            if (!batchText.trim()) return [];

            const prompt = `
            Analyze the following ${conversations.length} conversations and determine if they are related to the topic: "${topic}".
            For each conversation, determine:
            1. If it's relevant (YES/NO)
            2. A relevance score (0-10)
            
            Return the results in exactly this format, one line per conversation:
            CONV1: YES/NO: SCORE
            CONV2: YES/NO: SCORE
            etc.
            
            Conversations:
            ${batchText}
            `;

            const result = await this.model.generateContent(prompt);
            const resultText = result.response.text().trim();
            
            return resultText.split('\n').map(line => {
                const isRelevant = line.includes("YES");
                const scoreMatch = line.match(/:\s*(\d+)/);
                const relevanceScore = scoreMatch ? parseInt(scoreMatch[1], 10) : 5;

                return { isRelevant, relevanceScore };
            });
        } catch (error) {
            console.error('Error analyzing batch with AI:', error);
            // Return neutral results on error
            return conversations.map(() => ({ isRelevant: false, relevanceScore: 0 }));
        }
    }

    public static fallbackKeywordSearch(conversations: Message[][], topic: string): ConversationWithContext[] {
        return conversations
            .filter(conversation => 
                conversation.some(msg => 
                    msg.content.toLowerCase().includes(topic.toLowerCase())
                )
            )
            .map(conversation => ({
                messages: conversation,
                relevanceScore: 5
            }));
    }
}
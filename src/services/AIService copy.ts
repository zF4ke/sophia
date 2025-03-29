import { GoogleGenerativeAI } from "@google/generative-ai";
import { Message, ChatInputCommandInteraction } from "discord.js";
import { ConversationWithContext, AIAnalysisResult } from "../types/conversation";
import { EMOJIS } from "../utils/constants";

export class AIService {
    private static genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");
    private static model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    private static calculateBatchSize(totalConversations: number, accuracy: "normal" | "high"): number {
        if (accuracy === "high") {
            if (totalConversations <= 10) return 5;
            if (totalConversations <= 50) return 10;
            if (totalConversations <= 100) return 20;
            return 30;
        }
        if (totalConversations <= 10) return 10;
        if (totalConversations <= 50) return 25;
        if (totalConversations <= 100) return 50;
        if (totalConversations <= 500) return 100;
        return 200;
    }

    public static async analyzeConversations(
        conversations: Message[][],
        topic: string,
        interaction: ChatInputCommandInteraction,
        accuracy: "normal" | "high" = "normal"
    ): Promise<ConversationWithContext[]> {
        const relevantConversations: ConversationWithContext[] = [];
        const batchSize = this.calculateBatchSize(conversations.length, accuracy);
        const topicKeywords = this.extractKeywords(topic);
        let secondaryConversations: ConversationWithContext[] = [];

        for (let i = 0; i < conversations.length; i += batchSize) {
            const batch = conversations.slice(i, i + batchSize);
            const batchResults = await this.analyzeBatch(batch, topic, topicKeywords, accuracy);

            if (i % (batchSize * 2) === 0) {
                await interaction.editReply(
                    `${EMOJIS.search} Analisando conversas... (${i}/${conversations.length} processadas)`
                );
            }

            batch.forEach((conversation, index) => {
                const result = batchResults[index];
                if (!result) return;

                const minRelevanceScore = accuracy === "high" ? 4 : 3;
                const minWordCount = accuracy === "high" ? 12 : 8;
                const totalWords = conversation.reduce((sum, msg) => 
                    sum + msg.content.split(/\s+/).length, 0);

                // For high accuracy, store lower relevance conversations separately
                if (accuracy === "high" && result.relevanceScore >= 2 && totalWords >= 8) {
                    secondaryConversations.push({
                        messages: conversation,
                        relevanceScore: result.relevanceScore
                    });
                }

                if (result.relevanceScore >= minRelevanceScore && totalWords >= minWordCount) {
                    relevantConversations.push({
                        messages: conversation,
                        relevanceScore: result.relevanceScore
                    });
                }
            });
        }

        const validConversations = relevantConversations
            .filter(conv => this.validateRelevance(conv, topic, accuracy))
            .sort((a, b) => b.relevanceScore - a.relevanceScore);

        // If in high accuracy mode and few primary results, include secondary results
        if (accuracy === "high" && validConversations.length < 3) {
            const validSecondaryConversations = secondaryConversations
                .filter(conv => this.validateSecondaryRelevance(conv, topic))
                .sort((a, b) => b.relevanceScore - a.relevanceScore)
                .slice(0, 10); // Limit secondary results

            return [...validConversations, ...validSecondaryConversations];
        }

        return validConversations;
    }

    private static validateRelevance(
        conversation: ConversationWithContext, 
        topic: string,
        accuracy: "normal" | "high"
    ): boolean {
        const topicLower = topic.toLowerCase();
        const conversationText = conversation.messages
            .map(msg => msg.content.toLowerCase())
            .join(' ');

        const hasDirectMention = conversationText.includes(topicLower);
        const keywordMatches = this.extractKeywords(topic)
            .filter(kw => conversationText.includes(kw))
            .length;
        
        const minMessageLength = accuracy === "high" ? 12 : 8;
        const hasHighEngagement = conversation.messages.length >= (accuracy === "high" ? 3 : 2) &&
            conversation.messages.some(msg => msg.content.split(/\s+/).length > minMessageLength);

        if (accuracy === "high") {
            return (hasDirectMention && conversation.relevanceScore >= 4) ||
                   (keywordMatches >= 2 && conversation.relevanceScore >= 5) ||
                   (hasHighEngagement && keywordMatches >= 1 && conversation.relevanceScore >= 6) ||
                   (conversation.relevanceScore >= 8);
        }

        return (hasDirectMention && conversation.relevanceScore >= 3) ||
               (keywordMatches >= 2 && conversation.relevanceScore >= 4) ||
               (hasHighEngagement && keywordMatches >= 1 && conversation.relevanceScore >= 5) ||
               (conversation.relevanceScore >= 7);
    }

    private static validateSecondaryRelevance(
        conversation: ConversationWithContext,
        topic: string
    ): boolean {
        const conversationText = conversation.messages
            .map(msg => msg.content.toLowerCase())
            .join(' ');

        const hasAnyKeywords = this.extractKeywords(topic)
            .some(kw => conversationText.includes(kw));
        
        const hasSubstantialContent = conversation.messages.length >= 2 &&
            conversation.messages.some(msg => msg.content.split(/\s+/).length > 8);

        return (hasAnyKeywords && conversation.relevanceScore >= 2) ||
               (hasSubstantialContent && conversation.relevanceScore >= 3);
    }

    private static async analyzeBatch(
        conversations: Message[][], 
        topic: string,
        topicKeywords: string[],
        accuracy: "normal" | "high"
    ): Promise<AIAnalysisResult[]> {
        try {
            const batchText = conversations.map((conversation, index) => {
                const cleanedConversation = conversation.map(msg => {
                    const content = msg.content.replace(/[\\`*_{}[\]()#+\-.!]/g, '');
                    return `${msg.author.username}: ${content}`;
                }).join("\n");
                return `[Conversation ${index + 1}]\n${cleanedConversation}`;
            }).join("\n---\n");

            if (!batchText.trim()) return [];

            const prompt = accuracy === "high" ? this.getHighAccuracyPrompt(topic, batchText) : this.getNormalPrompt(topic, batchText);

            const result = await this.model.generateContent(prompt);
            const resultText = result.response.text().trim();
            
            return resultText.split('\n').map(line => {
                const scoreMatch = line.match(/(\d+)\/10/);
                const detailScores = line.match(/topic:(\d+) context:(\d+) quality:(\d+)/);
                
                let relevanceScore = 0;
                if (scoreMatch) {
                    relevanceScore = parseInt(scoreMatch[1], 10);
                } else if (detailScores) {
                    relevanceScore = Math.min(10, 
                        parseInt(detailScores[1], 10) + 
                        parseInt(detailScores[2], 10) + 
                        parseInt(detailScores[3], 10)
                    );
                }

                return {
                    isRelevant: relevanceScore >= (accuracy === "high" ? 4 : 3),
                    relevanceScore
                };
            });
        } catch (error) {
            console.error('Error in AI analysis:', error);
            return conversations.map(conv => this.fallbackAnalysis(conv, topic, topicKeywords, accuracy));
        }
    }

    private static getNormalPrompt(topic: string, batchText: string): string {
        return `
        Analyze these conversations for relevance to: "${topic}"
        Balance between being thorough and not too strict.

        Score each conversation (max 10 points) based on:
        1. Direct Topic Match (0-4 points):
           - Exact topic mentions (4 points)
           - Key topic terms (2-3 points)
           - Topic synonyms/related terms (1-2 points)
        
        2. Context Relevance (0-3 points):
           - Direct topic discussion (3 points)
           - Related concepts/context (1-2 points)
           - Implicit references (1 point)
        
        3. Conversation Quality (0-3 points):
           - Meaningful discussion (2-3 points)
           - Multiple messages (1-2 points)
           - Information value (1 point)

        Guidelines:
        - Consider both explicit and implicit topic references
        - Value meaningful discussions over brief mentions
        - Consider conversation context and flow
        - Include related discussions that add value
        - Score 0 only if completely unrelated

        Format each response exactly as:
        CONV#: TOTAL/10 (topic:X context:Y quality:Z)
        Where X,Y,Z are individual scores and TOTAL is their sum.

        Conversations:
        ${batchText}
        `;
    }

    private static getHighAccuracyPrompt(topic: string, batchText: string): string {
        return `
        Perform a detailed analysis of these conversations regarding: "${topic}"
        Be thorough but maintain reasonable standards.

        Score each conversation (max 10 points) with detailed criteria:
        1. Topic Match & Relevance (0-4 points):
           - Direct topic discussion (4 points)
           - Key concept mentions (3 points)
           - Related terminology (2 points)
           - Contextual references (1 point)
        
        2. Context & Depth (0-3 points):
           - Deep topic exploration (3 points)
           - Contextual understanding (2 points)
           - Related themes/concepts (1 point)
        
        3. Conversation Quality (0-3 points):
           - Substantive exchange (3 points)
           - Multiple participant engagement (2 points)
           - Clear information value (1 point)

        Required Analysis Steps:
        1. Identify all topic-related keywords and phrases
        2. Evaluate conversation depth and context
        3. Assess information quality and relevance
        4. Consider conversation flow and coherence
        5. Verify meaningful participant engagement
        6. Check for factual/informative content

        Scoring Guidelines:
        - Require concrete topic connections
        - Value depth over surface mentions
        - Consider conversation context heavily
        - Prioritize informative exchanges
        - Be strict with relevance scores

        Format each response exactly as:
        CONV#: TOTAL/10 (topic:X context:Y quality:Z)
        Where X,Y,Z are individual scores and TOTAL is their sum.

        Conversations:
        ${batchText}
        `;
    }

    private static extractKeywords(topic: string): string[] {
        const commonWords = new Set(['e', 'ou', 'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas', 
            'um', 'uma', 'uns', 'umas', 'o', 'a', 'os', 'as', 'para', 'por', 'com', 'sem']);
        
        return topic
            .toLowerCase()
            .split(/[\s,.-]+/)
            .filter(word => word.length > 2 && !commonWords.has(word));
    }

    private static fallbackAnalysis(
        conversation: Message[], 
        topic: string, 
        topicKeywords: string[],
        accuracy: "normal" | "high"
    ): AIAnalysisResult {
        const conversationText = conversation.map(msg => 
            msg.content.toLowerCase().replace(/[\\`*_{}[\]()#+\-.!]/g, '')
        ).join(' ');

        let score = 0;
        const topicLower = topic.toLowerCase();

        if (conversationText.includes(topicLower)) {
            score += accuracy === "high" ? 5 : 4;
        }

        const keywordMatches = topicKeywords.filter(kw => conversationText.includes(kw)).length;
        score += Math.min(accuracy === "high" ? 4 : 3, keywordMatches * (accuracy === "high" ? 2 : 1));

        const minWords = accuracy === "high" ? 10 : 6;
        const meaningfulMessages = conversation.filter(msg => {
            const words = msg.content.split(/\s+/).length;
            return words > minWords;
        }).length;
        
        score += Math.min(3, meaningfulMessages / (accuracy === "high" ? 1.5 : 2));

        const minMessages = accuracy === "high" ? 3 : 2;
        if (conversation.length >= minMessages && meaningfulMessages >= minMessages) {
            score += accuracy === "high" ? 2 : 1;
        }

        const minRelevanceScore = accuracy === "high" ? 4 : 3;
        const isRelevant = score >= minRelevanceScore && (
            conversationText.includes(topicLower) || 
            (keywordMatches >= (accuracy === "high" ? 3 : 2)) || 
            (keywordMatches >= 1 && meaningfulMessages >= (accuracy === "high" ? 3 : 2)) ||
            (meaningfulMessages >= (accuracy === "high" ? 4 : 3))
        );

        return {
            isRelevant,
            relevanceScore: Math.min(10, score)
        };
    }

    public static fallbackKeywordSearch(
        conversations: Message[][], 
        topic: string,
        accuracy: "normal" | "high" = "normal"
    ): ConversationWithContext[] {
        const topicKeywords = this.extractKeywords(topic);
        const allResults = conversations
            .map(conversation => {
                const analysis = this.fallbackAnalysis(conversation, topic, topicKeywords, accuracy);
                return {
                    messages: conversation,
                    relevanceScore: analysis.relevanceScore
                };
            })
            .sort((a, b) => b.relevanceScore - a.relevanceScore);

        // For high accuracy, first try strict filtering
        if (accuracy === "high") {
            const strictResults = allResults.filter(conv => {
                const minWords = 12;
                const hasEnoughContent = conv.messages.reduce((sum, msg) => 
                    sum + msg.content.split(/\s+/).length, 0) >= minWords;
                return conv.relevanceScore >= 4 && hasEnoughContent;
            });

            // If few strict results, include more lenient ones
            if (strictResults.length < 3) {
                const lenientResults = allResults.filter(conv => {
                    const hasEnoughContent = conv.messages.reduce((sum, msg) => 
                        sum + msg.content.split(/\s+/).length, 0) >= 8;
                    return conv.relevanceScore >= 2 && hasEnoughContent;
                }).slice(0, 10);

                return [...strictResults, ...lenientResults];
            }

            return strictResults;
        }

        // Normal accuracy uses original filtering
        return allResults.filter(conv => {
            const hasEnoughContent = conv.messages.reduce((sum, msg) => 
                sum + msg.content.split(/\s+/).length, 0) >= 8;
            return conv.relevanceScore >= 3 && hasEnoughContent;
        });
    }
}
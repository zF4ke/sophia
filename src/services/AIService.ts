import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import { Message, ChatInputCommandInteraction } from "discord.js";
import { ConversationWithContext, AIAnalysisResult } from "../types/conversation";
import { EMOJIS } from "../utils/constants";

export class AIService {
    private static genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");
    private static model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    
    private static contextModel = this.genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
            temperature: 0.4,
            topP: 0.8,
            topK: 40,
            maxOutputTokens: 2048,
        }
    });

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
        const topicKeywords = this.extractKeywords(topic);

        for (let i = 0; i < conversations.length; i += batchSize) {
            const batch = conversations.slice(i, i + batchSize);
            const batchResults = await this.analyzeBatch(batch, topic, topicKeywords);

            if (i % (batchSize * 2) === 0) {
                await interaction.editReply(
                    `${EMOJIS.search} Analisando conversas... (${i}/${conversations.length} processadas)`
                );
            }

            batch.forEach((conversation, index) => {
                const result = batchResults[index];
                if (result?.relevanceScore >= 3) {
                    const totalWords = conversation.reduce((sum, msg) => 
                        sum + msg.content.split(/\s+/).length, 0);
                    
                    if (totalWords >= 8) {
                        relevantConversations.push({
                            messages: conversation,
                            relevanceScore: result.relevanceScore
                        });
                    }
                }
            });
        }

        return relevantConversations
            .filter(conv => this.validateRelevance(conv, topic))
            .sort((a, b) => b.relevanceScore - a.relevanceScore);
    }

    private static validateRelevance(conversation: ConversationWithContext, topic: string): boolean {
        const topicLower = topic.toLowerCase();
        const conversationText = conversation.messages
            .map(msg => msg.content.toLowerCase())
            .join(' ');

        const hasDirectMention = conversationText.includes(topicLower);
        const keywordMatches = this.extractKeywords(topic)
            .filter(kw => conversationText.includes(kw))
            .length;
        const hasHighEngagement = conversation.messages.length >= 2 &&
            conversation.messages.some(msg => msg.content.split(/\s+/).length > 8);

        return (hasDirectMention && conversation.relevanceScore >= 3) ||
               (keywordMatches >= 2 && conversation.relevanceScore >= 4) ||
               (hasHighEngagement && keywordMatches >= 1 && conversation.relevanceScore >= 5) ||
               (conversation.relevanceScore >= 7);
    }

    private static async analyzeBatch(
        conversations: Message[][], 
        topic: string,
        topicKeywords: string[]
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

            const prompt = `
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
                    isRelevant: relevanceScore >= 3,
                    relevanceScore
                };
            });
        } catch (error) {
            console.error('Error in AI analysis:', error);
            return conversations.map(conv => this.fallbackAnalysis(conv, topic, topicKeywords));
        }
    }

    private static extractKeywords(text: string): string[] {
        const commonWords = new Set([
            'e', 'ou', 'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas', 
            'um', 'uma', 'uns', 'umas', 'o', 'a', 'os', 'as', 'para', 'por', 'com', 'sem',
            'the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'from', 'by', 'about'
        ]);
        
        return text
            .toLowerCase()
            .split(/[\s,.-]+/)
            .filter(word => word.length > 2 && !commonWords.has(word));
    }

    private static fallbackAnalysis(conversation: Message[], topic: string, topicKeywords: string[]): AIAnalysisResult {
        const conversationText = conversation.map(msg => 
            msg.content.toLowerCase().replace(/[\\`*_{}[\]()#+\-.!]/g, '')
        ).join(' ');

        let score = 0;
        const topicLower = topic.toLowerCase();

        if (conversationText.includes(topicLower)) {
            score += 4;
        }

        const keywordMatches = topicKeywords.filter(kw => conversationText.includes(kw)).length;
        score += Math.min(3, keywordMatches);

        const meaningfulMessages = conversation.filter(msg => {
            const words = msg.content.split(/\s+/).length;
            return words > 6;
        }).length;
        
        score += Math.min(3, meaningfulMessages / 2);

        if (conversation.length >= 2 && meaningfulMessages >= 2) {
            score += 1;
        }

        const isRelevant = score >= 3 && (
            conversationText.includes(topicLower) || 
            (keywordMatches >= 2) || 
            (keywordMatches >= 1 && meaningfulMessages >= 2) ||
            (meaningfulMessages >= 3)
        );

        return {
            isRelevant,
            relevanceScore: Math.min(10, score)
        };
    }

    public static fallbackKeywordSearch(conversations: Message[][], topic: string): ConversationWithContext[] {
        const topicKeywords = this.extractKeywords(topic);
        
        return conversations
            .map(conversation => {
                const analysis = this.fallbackAnalysis(conversation, topic, topicKeywords);
                return {
                    messages: conversation,
                    relevanceScore: analysis.relevanceScore
                };
            })
            .filter(conv => {
                const hasEnoughContent = conv.messages.reduce((sum, msg) => 
                    sum + msg.content.split(/\s+/).length, 0) >= 8;
                return conv.relevanceScore >= 3 && hasEnoughContent;
            })
            .sort((a, b) => b.relevanceScore - a.relevanceScore);
    }

    public static selectConversationsForContext(
        conversations: Message[][], 
        prompt: string, 
        maxChars: number = 50000
    ): Message[][] {
        const promptKeywords = this.extractKeywords(prompt);
        
        const scoredConversations = conversations.map(conversation => {
            const combinedText = conversation.map(msg => msg.content).join(' ').toLowerCase();
            
            const keywordScore = promptKeywords.reduce((score, keyword) => {
                return score + (combinedText.includes(keyword.toLowerCase()) ? 1 : 0);
            }, 0);
            
            const lengthScore = Math.min(5, conversation.length / 2);
            
            const charLength = conversation.reduce((sum, msg) => 
                sum + msg.author.username.length + 2 + msg.content.length, 0);
            
            return {
                conversation,
                score: keywordScore * 2 + lengthScore,
                charLength
            };
        });
        
        scoredConversations.sort((a, b) => b.score - a.score);
        
        const selected: Message[][] = [];
        let totalChars = 0;
        
        for (const item of scoredConversations) {
            if (totalChars + item.charLength <= maxChars) {
                selected.push(item.conversation);
                totalChars += item.charLength;
            } else {
                if (selected.length === 0) {
                    selected.push(item.conversation);
                }
                break;
            }
        }
        
        return selected;
    }

    public static formatConversationsAsContext(conversations: Message[][]): string {
        return conversations.map((conversation, index) => {
            const formattedConversation = conversation.map(msg => 
                `${msg.author.username}: ${msg.content.trim()}`
            ).join('\n');
            
            return `[Conversa ${index + 1}]\n${formattedConversation}`;
        }).join('\n\n');
    }

    public static async generateContextualResponse(prompt: string, context: string): Promise<string> {
        try {
            const aiPrompt = `
            Você tem acesso a conversas de um canal do Discord. Use essas conversas como contexto para responder à pergunta ou executar a instrução do usuário.
            
            Contexto das conversas:
            ${context}
            
            Pergunta/instrução do usuário:
            ${prompt}
            
            Diretrizes:
            - Base sua resposta no contexto fornecido
            - Se o contexto não contiver informações relevantes, diga isso claramente
            - Cite partes específicas do contexto para justificar sua resposta quando relevante
            - Seja conciso mas completo
            - Formate sua resposta de forma clara e organizada
            - Não inclua prefixos como "Baseado no contexto" ou "Resposta:"
            `;
            
            const result = await this.contextModel.generateContent(aiPrompt);
            const response = result.response.text();
            
            return response;
        } catch (error) {
            console.error('Error generating AI response:', error);
            return "Desculpe, não foi possível gerar uma resposta com base no contexto fornecido. Ocorreu um erro ao processar a solicitação.";
        }
    }
}
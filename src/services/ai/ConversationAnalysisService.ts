import { Message, ChatInputCommandInteraction } from "discord.js";
import { AIBaseService } from "./AIBaseService";
import { TextProcessingService } from "./TextProcessingService";
import { ConversationWithContext, AIAnalysisResult } from "../../types/conversation";
import { EMOJIS } from "../../utils/constants";

/**
 * Service for analyzing conversations and determining their relevance to topics
 */
export class ConversationAnalysisService extends AIBaseService {
  /**
   * Calculates optimal batch size based on total number of conversations
   * @param totalConversations - Total number of conversations to analyze
   * @returns Appropriate batch size for processing
   * @private
   */
  private static calculateBatchSize(totalConversations: number): number {
    if (totalConversations <= 10) return 10;
    if (totalConversations <= 50) return 25;
    if (totalConversations <= 100) return 50;
    if (totalConversations <= 500) return 100;
    return 200;
  }

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
    const relevantConversations: ConversationWithContext[] = [];
    const batchSize = this.calculateBatchSize(conversations.length);
    const topicKeywords = TextProcessingService.extractKeywords(topic);

    for (let i = 0; i < conversations.length; i += batchSize) {
      const batch = conversations.slice(i, i + batchSize);
      const batchResults = await this.analyzeBatch(batch, topic, topicKeywords);

      // Update progress periodically
      if (i % (batchSize * 2) === 0) {
        await interaction.editReply(
          `${EMOJIS.search} Analisando conversas... (${i}/${conversations.length} processadas)`
        );
      }

      // Filter and add relevant conversations
      batch.forEach((conversation, index) => {
        const result = batchResults[index];
        if (result?.relevanceScore >= 3) {
          const totalWords = conversation.reduce(
            (sum, msg) => sum + msg.content.split(/\s+/).length, 0
          );
          
          // Ensure conversation has enough content
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

  /**
   * Validates whether a conversation is truly relevant to the topic
   * @param conversation - Conversation with context and initial relevance score
   * @param topic - Topic to validate relevance against
   * @returns Boolean indicating if conversation passes additional relevance checks
   * @private
   */
  private static validateRelevance(
    conversation: ConversationWithContext, 
    topic: string
  ): boolean {
    const topicLower = topic.toLowerCase();
    const conversationText = conversation.messages
      .map(msg => msg.content.toLowerCase())
      .join(' ');

    const hasDirectMention = conversationText.includes(topicLower);
    const keywordMatches = TextProcessingService.extractKeywords(topic)
      .filter(kw => conversationText.includes(kw))
      .length;
    const hasHighEngagement = conversation.messages.length >= 2 &&
      conversation.messages.some(msg => msg.content.split(/\s+/).length > 8);

    // Apply tiered validation based on different criteria
    return (hasDirectMention && conversation.relevanceScore >= 3) ||
           (keywordMatches >= 2 && conversation.relevanceScore >= 4) ||
           (hasHighEngagement && keywordMatches >= 1 && conversation.relevanceScore >= 5) ||
           (conversation.relevanceScore >= 7);
  }

  /**
   * Analyzes a batch of conversations using AI to determine relevance to a topic
   * @param conversations - Batch of conversations to analyze
   * @param topic - Topic to analyze against
   * @param topicKeywords - Pre-extracted keywords from the topic
   * @returns Array of analysis results with relevance scores
   * @private
   */
  private static async analyzeBatch(
    conversations: Message[][], 
    topic: string,
    topicKeywords: string[]
  ): Promise<AIAnalysisResult[]> {
    try {
      // Format conversations for AI analysis
      const batchText = conversations.map((conversation, index) => {
        const cleanedConversation = conversation.map(msg => {
          const content = TextProcessingService.cleanText(msg.content);
          return `${msg.author.username}: ${content}`;
        }).join("\n");
        return `[Conversation ${index + 1}]\n${cleanedConversation}`;
      }).join("\n---\n");

      if (!batchText.trim()) return [];

      // Create prompt for AI analysis
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

      // Generate AI analysis
      const result = await this.defaultModel.generateContent(prompt);
      const resultText = result.response.text().trim();
      
      // Parse AI response for scores
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
      console.error('Error in AI conversation analysis:', error);
      // Fall back to keyword-based analysis if AI fails
      return conversations.map(conv => this.fallbackAnalysis(conv, topic, topicKeywords));
    }
  }

  /**
   * Performs fallback analysis when AI analysis fails
   * @param conversation - The conversation to analyze
   * @param topic - Topic to analyze against
   * @param topicKeywords - Pre-extracted keywords from the topic
   * @returns Analysis result with relevance score
   * @private
   */
  private static fallbackAnalysis(
    conversation: Message[], 
    topic: string, 
    topicKeywords: string[]
  ): AIAnalysisResult {
    const conversationText = conversation.map(msg => 
      msg.content.toLowerCase().replace(/[\\`*_{}[\]()#+\-.!]/g, '')
    ).join(' ');

    let score = 0;
    const topicLower = topic.toLowerCase();

    // Direct topic mention (highest relevance)
    if (conversationText.includes(topicLower)) {
      score += 4;
    }

    // Keyword matches
    const keywordMatches = topicKeywords.filter(kw => conversationText.includes(kw)).length;
    score += Math.min(3, keywordMatches);

    // Content quality assessment
    const meaningfulMessages = conversation.filter(msg => {
      const words = msg.content.split(/\s+/).length;
      return words > 6;
    }).length;
    
    score += Math.min(3, meaningfulMessages / 2);

    // Engagement bonus
    if (conversation.length >= 2 && meaningfulMessages >= 2) {
      score += 1;
    }

    // Determine if conversation is relevant based on composite criteria
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
    const topicKeywords = TextProcessingService.extractKeywords(topic);
    
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
}
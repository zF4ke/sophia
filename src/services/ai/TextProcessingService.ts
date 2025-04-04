import { DISCORD } from "@/utils/constants";
import { AIBaseService } from "./AIBaseService";
import dedent from "dedent";
import { Client } from "discord.js";

/**
 * Service for text processing operations like keyword extraction and text cleaning
 */
export class TextProcessingService extends AIBaseService {
    /**
     * Set of common words to exclude from keyword extraction in multiple languages
     * @private
     */
    private static readonly COMMON_WORDS = new Set([
        // Portuguese stopwords
        'e', 'ou', 'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas',
        'um', 'uma', 'uns', 'umas', 'o', 'a', 'os', 'as', 'para', 'por', 'com', 'sem',
        'que', 'como', 'mais', 'mas', 'já', 'ao', 'esta', 'este', 'esse', 'isso',

        // English stopwords
        'the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'from', 'by', 'about',
        'as', 'is', 'it', 'be', 'this', 'that', 'are', 'was', 'were', 'has', 'have'
    ]);

    /**
     * Extracts meaningful keywords from a text by removing common words and short terms
     * @param text - The input text to extract keywords from
     * @param minLength - Minimum length for a keyword (default: 3)
     * @returns Array of extracted keywords
     */
    public static extractKeywords(text: string, minLength: number = 3): string[] {
        return text
            .toLowerCase()
            .split(/[\s,.-]+/)
            .filter(word => word.length >= minLength && !this.COMMON_WORDS.has(word));
    }

    /**
     * Cleans text by removing Markdown formatting characters
     * @param text - The text to clean
     * @returns Cleaned text without markdown characters
     */
    public static cleanText(text: string): string {
        return text.replace(/[\\`*_{}[\]()#+\-.!]/g, '').trim();
    }

    /**
     * Adds author information to a question for better context
     * @param question - The question to format
     * @param author - The author of the question
     * @returns Contextualized question with author information
     */
    public static addAuthorToQuestion(question: string, author: string): string {
        return `[Mensagem de ${author}]: ${question}`
    }

    /**
     * Splits a long message into chunks that fit within Discord's message limit
     * @param message The message to split
     * @param limit The maximum length per chunk (default: Discord's message limit)
     * @returns Array of message chunks
     */
    public static splitLongMessage(message: string, limit: number = DISCORD.MESSAGE_LIMIT): string[] {
        const chunks: string[] = [];

        // If message is already within limit, return it as is
        if (message.length <= limit) {
            return [message];
        }

        let currentChunk = '';
        // Split by paragraphs (double newlines) first to maintain logical structure
        const paragraphs = message.split('\n\n');

        for (const paragraph of paragraphs) {
            // If adding this paragraph would exceed the limit, push current chunk and start a new one
            if ((currentChunk + paragraph + '\n\n').length > limit) {
                // If the paragraph itself is too long, split it further
                if (paragraph.length > limit) {
                    // First push current chunk if it exists
                    if (currentChunk) {
                        chunks.push(currentChunk);
                        currentChunk = '';
                    }

                    // Split long paragraph by sentences and try to keep sentences together
                    const sentences = paragraph.split(/(?<=\.|\?|\!) /);
                    for (const sentence of sentences) {
                        if ((currentChunk + sentence + ' ').length <= limit) {
                            currentChunk += sentence + ' ';
                        } else {
                            // If the sentence itself is too long, split by words
                            if (sentence.length > limit) {
                                if (currentChunk) {
                                    chunks.push(currentChunk);
                                    currentChunk = '';
                                }

                                // Split by words
                                let words = sentence.split(' ');
                                for (const word of words) {
                                    if ((currentChunk + word + ' ').length <= limit) {
                                        currentChunk += word + ' ';
                                    } else {
                                        chunks.push(currentChunk);
                                        currentChunk = word + ' ';
                                    }
                                }
                            } else {
                                chunks.push(currentChunk);
                                currentChunk = sentence + ' ';
                            }
                        }
                    }
                } else {
                    // Paragraph fits in a new chunk
                    chunks.push(currentChunk);
                    currentChunk = paragraph + '\n\n';
                }
            } else {
                // Add paragraph to current chunk
                currentChunk += paragraph + '\n\n';
            }
        }

        // Add the last chunk if it's not empty
        if (currentChunk) {
            chunks.push(currentChunk);
        }

        return chunks;
    }

    /**
     * Converts mentions in a message to usernames
     * @param message - The message containing mentions
     * @param client - The Discord client instance to fetch user information
     * @returns The message with mentions replaced by usernames
     */
    public static async convertMentionsToNames(message: string, client: Client): Promise<string> {
        const mentionRegex = /<@!?(\d+)>/g;
        const mentions = message.match(mentionRegex);
        if (!mentions) return message;

        for (const mention of mentions) {
            const userId = mention.replace(/<@!?/, '').replace(/>/, '');
            const user = await client.users.fetch(userId).catch(() => null);
            if (!user) continue;

            if (user) {
                message = message.replace(mention, user.username);
            }
        }

        return message;
    }

    public static removeMessageHeader(message: string): string {
        // example of a header: `✨ Resposta com conhecimento próprio`. it needs to have the emoji ✨ at the start of the line, but the text "Resposta com conhecimento próprio" could be anything. also the backticks are also part of the header.
        const headerRegex = /`✨.*?`/g;
        return message.replace(headerRegex, '').trim();
    }
}
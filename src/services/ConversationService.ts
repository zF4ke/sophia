import { Message } from "discord.js";
import { ConversationWithContext } from "../types/conversation";

export class ConversationService {
    private static readonly MAX_CONVERSATION_GAP = 5 * 60 * 1000;   // 5 minutes
    private static readonly SAME_AUTHOR_GAP = 8 * 60 * 1000;        // 8 minutes
    private static readonly REPLY_GAP = 10 * 60 * 1000;             // 10 minutes

    public static groupMessagesByConversation(messages: Message[]): Message[][] {
        const conversations: Message[][] = [];
        let currentConversation: Message[] = [];
        
        // Sort messages by timestamp (oldest first)
        const sortedMessages = [...messages].sort((a, b) => a.createdTimestamp - b.createdTimestamp);

        for (const message of sortedMessages) {
            if (!message.content.trim()) continue;

            if (!this.isRelatedToLastMessage(message, currentConversation)) {
                if (currentConversation.length > 0) {
                    conversations.push([...currentConversation]);
                    currentConversation = [];
                }
            }

            currentConversation.push(message);
        }

        if (currentConversation.length > 0) {
            conversations.push(currentConversation);
        }

        return conversations;
    }

    private static isRelatedToLastMessage(message: Message, currentConversation: Message[]): boolean {
        if (!currentConversation.length) return true;
        
        const lastMessage = currentConversation[currentConversation.length - 1];
        const timeDiff = message.createdTimestamp - lastMessage.createdTimestamp;
        
        // Check for direct replies
        if (message.reference?.messageId && 
            currentConversation.some(msg => msg.id === message.reference?.messageId)) {
            return timeDiff <= this.REPLY_GAP;
        }

        // Check for same author
        if (message.author.id === lastMessage.author.id) {
            return timeDiff <= this.SAME_AUTHOR_GAP;
        }

        // Regular messages
        return timeDiff <= this.MAX_CONVERSATION_GAP;
    }

    public static filterValidConversations(conversations: Message[][], includeBots: boolean = false): Message[][] {
        return conversations.filter(conversation => {
            // Filter out single short messages
            if (conversation.length === 1 && conversation[0].content.length < 10) return false;
            
            // Only filter out bot-only conversations if includeBots is false
            if (!includeBots && conversation.every(msg => msg.author.bot)) return false;
            
            return true;
        });
    }
}
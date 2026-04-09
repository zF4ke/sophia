import type { Message } from "discord.js";

export class ContextManagementService {
    public static selectConversationsForContext(
        conversations: Message[][],
        _prompt: string,
        _maxChars = 50000
    ): Message[][] {
        return conversations.slice(0, 8);
    }

    public static formatConversationsAsContext(conversations: Message[][]): string {
        return conversations
            .map((conversation, index) => {
                const text = conversation
                    .map((message) => `${message.author.username}: ${message.content}`)
                    .join("\n");
                return `[Conversa ${index + 1}]\n${text}`;
            })
            .join("\n\n");
    }

    public static formatMessagesAsContext(
        messages: Message[],
        includeBots = false,
        reverse = false
    ): string {
        const filtered = messages.filter((message) => includeBots || !message.author.bot);
        const ordered = reverse ? [...filtered].reverse() : filtered;
        return ordered.map((message) => `${message.author.username}: ${message.content}`).join("\n");
    }

    public static optimizeContextForTokenLimit(
        conversations: Message[][],
        prompt: string,
        _maxTokens = 8000
    ): string {
        return this.formatConversationsAsContext(
            this.selectConversationsForContext(conversations, prompt)
        );
    }
}

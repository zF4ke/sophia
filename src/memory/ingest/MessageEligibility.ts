import type { Message } from "discord.js";

export class MessageEligibility {
    public static isEligible(message: Message): boolean {
        if (!message.channel?.isTextBased()) {
            return false;
        }

        const content = message.content.trim();
        if (!content || content.startsWith("/")) {
            return false;
        }

        return !/^`[^\n]+`$/.test(content);
    }
}

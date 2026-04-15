import { MessageType, type Message } from "discord.js";

/** System message types that carry meaningful guild events. */
const ELIGIBLE_SYSTEM_TYPES: ReadonlySet<MessageType> = new Set([
    MessageType.UserJoin,
    MessageType.ChannelPinnedMessage,
    MessageType.ThreadCreated,
    MessageType.GuildBoost,
    MessageType.GuildBoostTier1,
    MessageType.GuildBoostTier2,
    MessageType.GuildBoostTier3,
]);

export class MessageEligibility {
    public static isEligible(message: Message): boolean {
        if (!message.channel?.isTextBased()) {
            return false;
        }

        // System messages (joins, pins, thread creation, boosts)
        if (ELIGIBLE_SYSTEM_TYPES.has(message.type)) {
            return true;
        }

        // Messages with embeds but no text content are still valuable
        if (message.embeds?.length > 0) {
            return true;
        }

        const content = message.content.trim();
        if (!content || content.startsWith("/")) {
            return false;
        }

        return !/^`[^\n]+`$/.test(content);
    }
}

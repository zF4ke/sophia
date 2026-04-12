import type { Message } from "discord.js";
import type { StoredMessage } from "@/memory/types";

export class MessageNormalizer {
    public static toStoredMessage(message: Message): StoredMessage {
        return {
            id: message.id,
            guildId: message.guildId || null,
            channelId: message.channelId,
            channelName:
                "name" in message.channel ? message.channel.name || message.channelId : message.channelId,
            authorId: message.author.id,
            authorName: message.member?.displayName || message.author.username,
            authorUsername: message.author.username,
            content: message.content.trim(),
            attachmentsJson: JSON.stringify(
                message.attachments.map((attachment) => ({
                    id: attachment.id,
                    name: attachment.name,
                    url: attachment.url,
                    contentType: attachment.contentType,
                }))
            ),
            referenceMessageId: message.reference?.messageId || null,
            createdTimestamp: message.createdTimestamp,
            jumpLink: message.url,
            isBot: message.author.bot ? 1 : 0,
        };
    }
}

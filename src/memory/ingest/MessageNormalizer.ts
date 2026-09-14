import { MessageType, type Message } from "discord.js";
import type { StoredMessage } from "@/memory/types";
import { extractComponentsText } from "@/memory/ingest/ComponentTextExtractor";
import { revisedMessageSource } from "@/shared/sourceReference";

/** Map system message types to human-readable descriptions. */
function describeSystemMessage(message: Message): string | null {
    switch (message.type) {
        case MessageType.UserJoin:
            return `[System] ${message.author.username} joined the server.`;
        case MessageType.ChannelPinnedMessage:
            return `[System] ${message.author.username} pinned a message.`;
        case MessageType.ThreadCreated:
            return `[System] ${message.author.username} started a thread: ${message.content || "unknown"}.`;
        case MessageType.GuildBoost:
            return `[System] ${message.author.username} boosted the server.`;
        case MessageType.GuildBoostTier1:
        case MessageType.GuildBoostTier2:
        case MessageType.GuildBoostTier3:
            return `[System] ${message.author.username} boosted the server to a new level!`;
        default:
            return null;
    }
}

/** Extract text content from embeds, if any. */
function extractEmbedContent(message: Message): string {
    if (!message.embeds?.length) return "";
    return message.embeds
        .map((embed) => {
            const parts: string[] = [];
            if (embed.title) parts.push(embed.title);
            if (embed.description) parts.push(embed.description);
            if (embed.fields?.length) {
                for (const field of embed.fields) {
                    parts.push(`${field.name}: ${field.value}`);
                }
            }
            return parts.join(" — ");
        })
        .filter(Boolean)
        .join("\n");
}

export class MessageNormalizer {
    public static toStoredMessage(message: Message): StoredMessage {
        const guildMember = message.member || message.guild?.members?.cache.get(message.author.id) || null;
        const authorDisplayName =
            guildMember?.displayName ||
            message.author.globalName ||
            message.author.username;

        // Build content from text, system description, embeds, and components
        const textContent = message.content.trim();
        const systemDesc = describeSystemMessage(message);
        const embedContent = extractEmbedContent(message);
        const componentsContent = extractComponentsText(message);

        const contentParts: string[] = [];
        if (systemDesc) contentParts.push(systemDesc);
        else if (textContent) contentParts.push(textContent);
        if (embedContent) contentParts.push(embedContent);
        if (componentsContent) contentParts.push(componentsContent);

        const stored: StoredMessage = {
            editedTimestamp: message.editedTimestamp,
            id: message.id,
            guildId: message.guildId || null,
            channelId: message.channelId,
            channelName:
                "name" in message.channel ? message.channel.name || message.channelId : message.channelId,
            authorId: message.author.id,
            authorName: authorDisplayName,
            authorUsername: message.author.username,
            authorNickname: guildMember?.nickname || null,
            content: contentParts.join("\n") || textContent,
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
        if (message.editedTimestamp) stored.jumpLink = revisedMessageSource(stored);
        return stored;
    }
}

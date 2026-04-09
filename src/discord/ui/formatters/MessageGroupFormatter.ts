import type { EmbedBuilder, Message } from "discord.js";
import type { MessageGroup } from "@/discord/ui/types";

const MAX_FIELD_VALUE_LENGTH = 1024;
const MAX_FIELD_NAME_LENGTH = 256;

export class MessageGroupFormatter {
    public static addMessageGroupFields(embed: EmbedBuilder, messages: Message[]): void {
        const messageGroups = this.groupMessagesByAuthor(messages);

        for (const group of messageGroups) {
            const date = new Date(group.timestamp).toLocaleDateString("pt-BR", {
                day: "2-digit",
                month: "2-digit",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit",
            });

            const baseFieldName = `${group.author} • ${date}`;
            const chunks: string[] = [];
            let currentChunk = "";

            for (const content of group.content) {
                if (!content.trim()) {
                    continue;
                }

                for (const part of this.splitContentIntoChunks(content)) {
                    if ((currentChunk + "\n" + part).length <= MAX_FIELD_VALUE_LENGTH) {
                        currentChunk = currentChunk ? `${currentChunk}\n${part}` : part;
                    } else {
                        if (currentChunk) {
                            chunks.push(currentChunk);
                        }
                        currentChunk = part;
                    }
                }
            }

            if (currentChunk) {
                chunks.push(currentChunk);
            }

            chunks.forEach((chunk, index) => {
                const fieldName =
                    chunks.length > 1
                        ? `${baseFieldName} (${index + 1}/${chunks.length})`
                        : baseFieldName;

                embed.addFields({
                    name:
                        fieldName.length > MAX_FIELD_NAME_LENGTH
                            ? `${fieldName.slice(0, MAX_FIELD_NAME_LENGTH - 3)}...`
                            : fieldName,
                    value: chunk.slice(0, MAX_FIELD_VALUE_LENGTH),
                });
            });
        }
    }

    private static groupMessagesByAuthor(messages: Message[]): MessageGroup[] {
        const groups = new Map<string, MessageGroup>();

        for (const message of messages) {
            const key = message.author.id;
            const content = message.content || "*Sem conteúdo*";

            if (!groups.has(key)) {
                groups.set(key, {
                    author: message.author.username,
                    content: [content],
                    timestamp: message.createdTimestamp,
                });
                continue;
            }

            const existing = groups.get(key)!;
            if (!existing.content.includes(content)) {
                existing.content.push(content);
            }
        }

        return [...groups.values()].sort((a, b) => a.timestamp - b.timestamp);
    }

    private static splitContentIntoChunks(content: string): string[] {
        if (content.length <= MAX_FIELD_VALUE_LENGTH) {
            return [content];
        }

        const chunks: string[] = [];
        let currentIndex = 0;
        while (currentIndex < content.length) {
            let endIndex = currentIndex + MAX_FIELD_VALUE_LENGTH;
            if (endIndex < content.length) {
                const lastSpace = content.lastIndexOf(" ", endIndex);
                if (lastSpace > currentIndex) {
                    endIndex = lastSpace;
                }
            }

            chunks.push(content.slice(currentIndex, endIndex).trim());
            currentIndex = endIndex;
        }

        return chunks;
    }
}

import { Message, TextChannel, ThreadChannel } from "discord.js";
import { MessageChunker } from "@/discord/ui/formatters/MessageChunker";

function isUnknownMessageReferenceError(error: unknown): boolean {
    if (!error || typeof error !== "object") {
        return false;
    }

    const apiError = error as {
        code?: unknown;
        rawError?: { errors?: { message_reference?: unknown } };
    };
    if (apiError.code !== 50035) {
        return false;
    }

    return Boolean(apiError.rawError?.errors?.message_reference);
}

export class ChannelMessenger {
    public static async sendLongMessage(message: Message, response: string): Promise<Message[]> {
        const channel = message.channel;
        if (!channel.isTextBased()) {
            return [];
        }
        if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) {
            return [];
        }

        if (!response.trim()) {
            return [];
        }

        const chunks = MessageChunker.split(response);
        const sent: Message[] = [];
        const firstChunk = chunks.shift() || "";

        let lastMessage: Message;
        try {
            lastMessage = await message.reply({ content: firstChunk });
        } catch (error) {
            if (!isUnknownMessageReferenceError(error)) {
                throw error;
            }
            // Fallback when the source message was deleted (e.g. clear_messages on current channel).
            lastMessage = await channel.send({ content: firstChunk });
        }
        sent.push(lastMessage);

        for (const chunk of chunks) {
            try {
                lastMessage = await lastMessage.reply({
                    content: chunk,
                    allowedMentions: {
                        parse: ["users"],
                        repliedUser: false,
                    },
                });
            } catch (error) {
                if (!isUnknownMessageReferenceError(error)) {
                    throw error;
                }
                lastMessage = await channel.send({
                    content: chunk,
                    allowedMentions: {
                        parse: ["users"],
                        repliedUser: false,
                    },
                });
            }
            sent.push(lastMessage);
        }

        return sent;
    }
}

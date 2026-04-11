import { Message, TextChannel, ThreadChannel } from "discord.js";
import { MessageChunker } from "@/discord/ui/formatters/MessageChunker";

export class ChannelMessenger {
    public static async sendLongMessage(message: Message, response: string): Promise<Message[]> {
        const channel = message.channel;
        if (!channel.isTextBased()) {
            return [];
        }
        if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) {
            return [];
        }

        const chunks = MessageChunker.split(response);
        const sent: Message[] = [];
        let lastMessage: Message | null = await message.reply({ content: chunks.shift() || "" });
        sent.push(lastMessage);

        for (const chunk of chunks) {
            lastMessage = await lastMessage.reply({
                content: chunk,
                allowedMentions: {
                    parse: ["users"],
                    repliedUser: false,
                },
            });
            sent.push(lastMessage);
        }

        return sent;
    }
}

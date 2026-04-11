import { Message, MessageFlags, type ChatInputCommandInteraction } from "discord.js";
import { MessageChunker } from "@/discord/ui/formatters/MessageChunker";

export class InteractionMessenger {
    public static async sendLongResponse(
        interaction: ChatInputCommandInteraction,
        header: string,
        response: string,
        ephemeral = false
    ): Promise<Message[]> {
        const content = header ? `${header}\n\n${response}` : response;
        const chunks = MessageChunker.split(content);
        const [firstChunk, ...rest] = chunks;
        const sent: Message[] = [];

        const firstMessage = await interaction.editReply({ content: firstChunk || "" });
        sent.push(firstMessage as Message);
        for (const chunk of rest) {
            const followUp = await interaction.followUp({
                content: chunk,
                flags: ephemeral ? MessageFlags.Ephemeral : undefined,
            });
            sent.push(followUp as Message);
        }

        return sent;
    }
}

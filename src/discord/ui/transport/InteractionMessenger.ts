import { MessageFlags, type ChatInputCommandInteraction } from "discord.js";
import { MessageChunker } from "@/discord/ui/formatters/MessageChunker";

export class InteractionMessenger {
    public static async sendLongResponse(
        interaction: ChatInputCommandInteraction,
        header: string,
        response: string,
        ephemeral = false
    ): Promise<void> {
        const content = header ? `${header}\n\n${response}` : response;
        const chunks = MessageChunker.split(content);
        const [firstChunk, ...rest] = chunks;

        await interaction.editReply({ content: firstChunk || "" });
        for (const chunk of rest) {
            await interaction.followUp({
                content: chunk,
                flags: ephemeral ? MessageFlags.Ephemeral : undefined,
            });
        }
    }
}

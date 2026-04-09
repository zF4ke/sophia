import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { AgentOrchestrator } from "@/agent/AgentOrchestrator";
import { SecurityService } from "@/security/SecurityService";
import { UIService } from "@/discord/ui/UIService";
import { EMOJIS } from "@/discord/constants";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("talk")
        .setDescription("Conversa com Sophia")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addStringOption((option) =>
            option
                .setName("message")
                .setDescription("Mensagem para Sophia")
                .setRequired(true)
        )
        .addBooleanOption((option) =>
            option
                .setName("ephemeral")
                .setDescription("Somente você pode ver a resposta")
                .setRequired(false)
        ),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        try {
            if (!SecurityService.isAdmin(interaction.user.id)) {
                await interaction.reply({
                    content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                    flags: MessageFlags.Ephemeral,
                });
                return;
            }

            const message = interaction.options.getString("message", true);
            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
            await interaction.deferReply({
                flags: ephemeral ? MessageFlags.Ephemeral : undefined,
            });

            const result = await AgentOrchestrator.answerQuestion({
                question: message,
                user: interaction.user,
                guild: interaction.guild,
                currentChannelId: interaction.channelId,
            });

            await UIService.sendLongResponse(
                interaction,
                "",
                UIService.formatAnswer(result.answer, result.citations),
                ephemeral
            );
        } catch (error) {
            console.error("Error in talk command:", error);
            await interaction.editReply(
                UIService.formatStatusMessage(
                    EMOJIS.error,
                    "Ocorreu um erro ao conversar.",
                    false
                )
            );
        }
    },
};

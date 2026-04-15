import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DebugService } from "@/discord/debug/DebugService";
import { ResponseActivityService } from "@/discord/responding/ResponseActivityIndicator";
import { UIService } from "@/discord/ui/UIService";
import { EMOJIS } from "@/discord/constants";
import { ConversationAdapter } from "@/discord/conversation/ConversationAdapter";
import { Runtime } from "@/runtime/Runtime";
import { SecurityService } from "@/security/SecurityService";
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
        let debugSession = null;
        let activityIndicator = null;

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
            activityIndicator = await ResponseActivityService.startForInteraction(interaction);
            await activityIndicator.startThinking();
            debugSession = await DebugService.startForInteraction(interaction, message);

            const input = await ConversationAdapter.fromInteraction({
                interaction,
                question: message,
                debugSession,
            });
            const result = await Runtime.answer(input);
            await activityIndicator.startTyping();
            const sentMessages = await UIService.sendLongResponse(
                interaction,
                "",
                UIService.formatAnswer(result.answer, result.citations),
                ephemeral
            );
            await ConversationAdapter.bindResponseMessages({
                input,
                result,
                sentMessages,
            });
        } catch (error) {
            console.error("Error in talk command:", error);
            await debugSession?.finishError(error);
            await interaction.editReply(
                UIService.formatStatusMessage(
                    EMOJIS.error,
                    "Ocorreu um erro ao conversar.",
                    false
                )
            );
        } finally {
            await activityIndicator?.stop();
        }
    },
};


import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { MessageService } from "../../services/MessageService";
import { ConversationService } from "../../services/ConversationService";
import { AIService } from "../../services/AIService";
import { EMOJIS, DISCORD } from "../../utils/constants";
import { SecurityService } from "../../services/SecurityService";
import { UIService } from "../../services/UIService";
import { TextProcessingService } from "@/services/ai/TextProcessingService";
import { ChattingService } from "@/services/ai/ChattingService";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("talk")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .setDescription("Usa mensagens do canal como contexto para falar")
        .addStringOption(option => 
            option.setName("message")
                .setDescription("A mensagem ou instrução para o AI")
                .setRequired(true))
        .addChannelOption(option => 
            option.setName("channel")
                .setDescription("O canal para usar como contexto")
                .setRequired(false))
        .addIntegerOption(option =>
            option.setName("limit")
                .setDescription("Número máximo de mensagens para buscar (padrão 0 para auto-cache)")
                .setMinValue(0)
                .setMaxValue(50000)
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("ephemeral")
                .setDescription("Apenas você pode ver a resposta (padrão: true)")
                .setRequired(false)),

    async execute(interaction: ChatInputCommandInteraction) {
        try {
            if (!SecurityService.isAdmin(interaction.user.id)) {
                return await interaction.reply({
                    content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                    flags: MessageFlags.Ephemeral
                });
            }

            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
            await interaction.deferReply({ flags: ephemeral ? MessageFlags.Ephemeral : undefined });

            const channel = interaction.options.getChannel("channel") || interaction.channel;
            const prompt = interaction.options.getString("message");
            const limit = interaction.options.getInteger("limit") ?? 100; // Get the limit if provided
            const includeBots = interaction.options.getBoolean("include_bots") ?? true;

            if (!channel || !prompt) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.info, "Por favor, forneça um canal e uma pergunta.", false));
            }

            if (!(channel instanceof TextChannel)) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "O canal deve ser um canal de texto.", false));
            }

            if (!channel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.error, "Eu não tenho permissão para ver esse canal.", false));
            }

            const response = await ChattingService.generateChatResponse(
                channel,
                prompt,
                {
                    limit,
                    includeBots,
                    userName: interaction.user.username,
                }
            );
            
            await UIService.sendLongResponse(
                interaction, 
                "", 
                response || "‎",
                ephemeral
            );
        } catch (error) {
            console.error('Error in context command:', error);
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.error, "Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente mais tarde.", false)
            );
        }
    },
};
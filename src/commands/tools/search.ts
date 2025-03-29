import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { MessageService } from "../../services/MessageService";
import { ConversationService } from "../../services/ConversationService";
import { AIService } from "../../services/AIService";
import { ConversationUIService } from "../../services/ui/ConversationUIService";
import { EMOJIS, ADMIN_IDS } from "../../utils/constants";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("search")
        .setDescription("Buscar mensagens em um canal sobre um tópico específico")
        .addChannelOption(option =>
            option.setName("channel")
                .setDescription("O canal para buscar")
                .setRequired(true))
        .addStringOption(option =>
            option.setName("topic")
                .setDescription("O tópico para buscar")
                .setRequired(true))
        .addIntegerOption(option =>
            option.setName("limit")
                .setDescription("Número máximo de mensagens para buscar (padrão: 2000)")
                .setMinValue(1)
                .setMaxValue(20000)
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("include_bots")
                .setDescription("Incluir mensagens de bots na busca (padrão: false)")
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("ephemeral")
                .setDescription("Apenas você pode ver o resultado da busca (padrão: false)")
                .setRequired(false)),

    async execute(interaction: ChatInputCommandInteraction) {
        try {
            // Check if user is admin
            if (!ADMIN_IDS.includes(interaction.user.id)) {
                return await interaction.reply({
                    content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                    ephemeral: true
                });
            }

            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
            await interaction.deferReply({ 
                flags: ephemeral ? MessageFlags.Ephemeral : undefined
            });

            const channel = interaction.options.getChannel("channel");
            const topic = interaction.options.getString("topic");
            const limit = interaction.options.getInteger("limit") || 2000;
            const includeBots = interaction.options.getBoolean("include_bots") ?? false;

            if (!channel || !topic) {
                return await interaction.editReply(`${EMOJIS.error} Por favor, forneça um canal e um tópico para buscar.`);
            }

            if (!(channel instanceof TextChannel)) {
                return await interaction.editReply(`${EMOJIS.error} O canal deve ser um canal de texto.`);
            }

            if (!channel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                return await interaction.editReply(`${EMOJIS.error} Eu não tenho permissão para ver esse canal.`);
            }

            await interaction.editReply(`${EMOJIS.search} Buscando mensagens sobre **"${topic}"** em ${channel}...`);

            try {
                // Fetch messages from the channel
                const messages = await MessageService.fetchMessages(channel, limit, interaction);

                if (messages.length === 0) {
                    return await interaction.editReply(`${EMOJIS.warning} Nenhuma mensagem encontrada no canal.`);
                }

                // Filter out command messages
                const filteredMessages = MessageService.filterCommandMessages(messages, interaction);

                // Group messages into conversations
                const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
                const validConversations = ConversationService.filterValidConversations(conversations, includeBots);

                // Analyze conversations with AI
                let relevantConversations;
                try {
                    relevantConversations = await AIService.analyzeConversations(validConversations, topic, interaction);
                } catch (error) {
                    console.error('AI analysis failed, falling back to keyword search:', error);
                    relevantConversations = AIService.fallbackKeywordSearch(validConversations, topic);
                }

                if (relevantConversations.length === 0) {
                    return await interaction.editReply(
                        `${EMOJIS.warning} Nenhuma mensagem relacionada a **"${topic}"** encontrada em ${channel}.`
                    );
                }

                // Display results with pagination, passing ephemeral flag
                await ConversationUIService.displayConversations(interaction, relevantConversations, topic, channel.name, ephemeral);

            } catch (error) {
                console.error('Error in search command:', error);
                throw error; // Re-throw to be caught by outer try-catch
            }

        } catch (error) {
            console.error('Error in search command:', error);
            await interaction.editReply(
                `${EMOJIS.error} Ocorreu um erro durante a busca. Por favor, tente novamente mais tarde.`
            );
        }
    },
};
import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { MessageService } from "../../services/MessageService";
import { ConversationService } from "../../services/ConversationService";
import { AIService } from "../../services/AIService";
import { ConversationUIService } from "../../services/ui/ConversationUIService";
import { EMOJIS } from "../../utils/constants";
import { SecurityService } from "../../services/SecurityService";
import { UIService } from "../../services/UIService";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("find")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
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
                .setDescription("Número máximo de mensagens para buscar (padrão 0 para auto-cache)")
                .setMinValue(0)
                .setMaxValue(50000)
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
            if (!SecurityService.isAdmin(interaction.user.id)) {
                return await interaction.reply({
                    content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                    flags: MessageFlags.Ephemeral
                });
            }

            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
            await interaction.deferReply({ 
                flags: ephemeral ? MessageFlags.Ephemeral : undefined
            });

            const channel = interaction.options.getChannel("channel");
            const topic = interaction.options.getString("topic");
            const providedLimit = interaction.options.getInteger("limit"); // Get the limit if provided
            const includeBots = interaction.options.getBoolean("include_bots") ?? false;

            if (!channel || !topic) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.info, "Por favor, forneça um canal e um tópico para buscar.", false));
            }

            if (!(channel instanceof TextChannel)) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "O canal deve ser um canal de texto.", false));
            }

            if (!channel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.error, "Eu não tenho permissão para ver esse canal.", false));
            }

            await interaction.editReply(UIService.formatStatusMessage(EMOJIS.loading, `Buscando mensagens sobre **"${topic}"** em ${channel}...`));

            try {
                // Use 0 as the limit if not provided, so MessageService will use cache size or fallback to 1000
                const limit = providedLimit || 0;
                
                // Fetch messages from the channel
                const messages = await MessageService.fetchMessages(channel, limit, interaction);

                if (messages.length === 0) {
                    return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "Nenhuma mensagem encontrada no canal.", false));
                }

                // Filter out command messages
                const filteredMessages = MessageService.filterCommandMessages(messages, interaction);

                // Group messages into conversations
                const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
                //const validConversations = ConversationService.filterValidConversations(conversations, includeBots);

                await interaction.editReply(UIService.formatStatusMessage(EMOJIS.found, `Analisando ${conversations.length} conversas sobre **"${topic}"**...`));

                // Analyze conversations with AI
                let relevantConversations;
                try {
                    relevantConversations = await AIService.analyzeConversations(conversations, topic, interaction);
                } catch (error) {
                    console.error('AI analysis failed, falling back to keyword search:', error);
                    await interaction.editReply(UIService.formatStatusMessage(EMOJIS.sync, "Usando busca alternativa por palavras-chave..."));
                    relevantConversations = AIService.fallbackKeywordSearch(conversations, topic);
                }

                if (relevantConversations.length === 0) {
                    return await interaction.editReply(
                        UIService.formatStatusMessage(EMOJIS.warning, `Nenhuma mensagem relacionada a **"${topic}"** encontrada em ${channel}.`, false)
                    );
                }

                await interaction.editReply(UIService.formatStatusMessage(EMOJIS.complete, `Encontradas ${relevantConversations.length} conversas relevantes!`));

                // Display results with pagination, passing ephemeral flag
                await ConversationUIService.displayConversations(interaction, relevantConversations, topic, channel.name, ephemeral);

            } catch (error) {
                console.error('Error in search command:', error);
                throw error; // Re-throw to be caught by outer try-catch
            }

        } catch (error) {
            console.error('Error in search command:', error);
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.error, "Ocorreu um erro durante a busca. Por favor, tente novamente mais tarde.", false)
            );
        }
    },
};
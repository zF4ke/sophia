import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { MessageService } from "../../services/MessageService";
import { ConversationService } from "../../services/ConversationService";
import { AIService } from "../../services/AIService";
import { EMOJIS, ADMIN_IDS } from "../../utils/constants";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("context")
        .setDescription("Usa mensagens do canal como contexto para uma pergunta")
        .addChannelOption(option => 
            option.setName("channel")
                .setDescription("O canal para usar como contexto")
                .setRequired(true))
        .addStringOption(option => 
            option.setName("prompt")
                .setDescription("A pergunta ou instrução para o AI")
                .setRequired(true))
        .addIntegerOption(option =>
            option.setName("limit")
                .setDescription("Número máximo de mensagens para buscar (padrão: 1000)")
                .setMinValue(1)
                .setMaxValue(10000)
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("include_bots")
                .setDescription("Incluir mensagens de bots no contexto (padrão: false)")
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("ephemeral")
                .setDescription("Apenas você pode ver a resposta (padrão: false)")
                .setRequired(false)),

    async execute(interaction: ChatInputCommandInteraction) {
        try {
            // Check if user is admin
            if (!ADMIN_IDS.includes(interaction.user.id)) {
                return await interaction.reply({
                    content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                    flags: MessageFlags.Ephemeral
                });
            }

            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
            await interaction.deferReply({ flags: MessageFlags.Ephemeral });

            const channel = interaction.options.getChannel("channel");
            const prompt = interaction.options.getString("prompt");
            const limit = interaction.options.getInteger("limit") || 1000;
            const includeBots = interaction.options.getBoolean("include_bots") ?? false;

            if (!channel || !prompt) {
                return await interaction.editReply(`${EMOJIS.error} Por favor, forneça um canal e uma pergunta.`);
            }

            if (!(channel instanceof TextChannel)) {
                return await interaction.editReply(`${EMOJIS.error} O canal deve ser um canal de texto.`);
            }

            if (!channel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                return await interaction.editReply(`${EMOJIS.error} Eu não tenho permissão para ver esse canal.`);
            }

            await interaction.editReply(`${EMOJIS.search} Buscando contexto em ${channel}...`);

            try {
                // Fetch messages using MessageService
                const messages = await MessageService.fetchMessages(channel, limit, interaction);

                if (messages.length === 0) {
                    return await interaction.editReply(`${EMOJIS.warning} Nenhuma mensagem encontrada no canal.`);
                }

                // Filter messages using MessageService
                const filteredMessages = MessageService.filterCommandMessages(messages, interaction);
                
                // Group messages into conversations using ConversationService
                const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
                const validConversations = ConversationService.filterValidConversations(conversations, includeBots);
                
                if (validConversations.length === 0) {
                    return await interaction.editReply(`${EMOJIS.warning} Não foi possível extrair conversas válidas do canal.`);
                }

                await interaction.editReply(`${EMOJIS.search} Processando ${validConversations.length} conversas como contexto...`);
                
                // Select relevant conversations and format them as context using AIService
                const selectedConversations = AIService.selectConversationsForContext(validConversations, prompt);
                const contextText = AIService.formatConversationsAsContext(selectedConversations);
                
                // Generate response using AIService
                const aiResponse = await AIService.generateContextualResponse(prompt, contextText);
                
                // Send the response
                await interaction.editReply({
                    content: `${EMOJIS.success} **Resposta baseada no contexto**\n\n**Sua pergunta:** ${prompt}\n\n**Resposta:**\n${aiResponse}`
                });

            } catch (error) {
                console.error('Error in context command:', error);
                throw error;
            }

        } catch (error) {
            console.error('Error in context command:', error);
            await interaction.editReply(
                `${EMOJIS.error} Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente mais tarde.`
            );
        }
    },
};
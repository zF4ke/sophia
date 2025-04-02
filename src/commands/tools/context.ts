import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { MessageService } from "../../services/MessageService";
import { ConversationService } from "../../services/ConversationService";
import { AIService } from "../../services/AIService";
import { EMOJIS, DISCORD } from "../../utils/constants";
import { SecurityService } from "../../services/SecurityService";
import { UIService } from "../../services/UIService";
import { TextProcessingService } from "@/services/ai/TextProcessingService";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("context")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .setDescription("Usa mensagens do canal como contexto para uma pergunta")
        .addStringOption(option => 
            option.setName("prompt")
                .setDescription("A pergunta ou instrução para o AI")
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
            option.setName("include_bots")
                .setDescription("Incluir mensagens de bots no contexto (padrão: false)")
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("ephemeral")
                .setDescription("Apenas você pode ver a resposta (padrão: false)")
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
            const prompt = interaction.options.getString("prompt");
            const limit = interaction.options.getInteger("limit") || 0;
            const includeBots = interaction.options.getBoolean("include_bots") ?? false;

            if (!channel || !prompt) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.info, "Por favor, forneça um canal e uma pergunta.", false));
            }

            if (!(channel instanceof TextChannel)) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "O canal deve ser um canal de texto.", false));
            }

            if (!channel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.error, "Eu não tenho permissão para ver esse canal.", false));
            }

            await interaction.editReply(UIService.formatStatusMessage(EMOJIS.loading, `Buscando contexto em \`**\`${channel.name}\`**\` ...`));

            const messages = await MessageService.fetchMessages(channel, limit, interaction);
            //console.log(`Fetched ${messages.length} messages from channel ${channel.name}`);
            if (messages.length === 0) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "Nenhuma mensagem encontrada no canal.", false));
            }

            // const filteredMessages = MessageService.filterCommandMessages(messages, interaction);~
            const filteredMessages = MessageService.filterOwnMessages(messages, interaction.client.user.id);
            const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
            // const validConversations = ConversationService.filterValidConversations(conversations, includeBots);
            
            // if (validConversations.length === 0) {
            //     return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "Não foi possível extrair conversas válidas do canal.", false));
            // }

            // await interaction.editReply(UIService.formatStatusMessage(EMOJIS.conversation, `Processando ${validConversations.length} conversas como contexto...`));

            if (conversations.length === 0) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "Nenhuma conversa válida encontrada no canal.", false));
            }

            await interaction.editReply(UIService.formatStatusMessage(EMOJIS.conversation, `Processando ${conversations.length} conversas como contexto...`));
            
            // const selectedConversations = AIService.selectConversationsForContext(validConversations, prompt);
            // const contextText = AIService.formatConversationsAsContext(selectedConversations);

            const contextText = AIService.formatConversationsAsContext(conversations);
            
            const promptWithAuthor = TextProcessingService.addAuthorToQuestion(prompt, interaction.user.username);
            const response = await AIService.generateContextualResponse(promptWithAuthor, contextText);
            const messageHeader = UIService.formatStatusMessage(EMOJIS.complete, `Resposta baseada no contexto`);
            
            await UIService.sendLongResponse(interaction, messageHeader, response, ephemeral);

            // console.log(`Context text length: ${contextText.length}`);
            // // token estimator, to estimate the number of tokens in the context text
            // // code it here
            // const tokenCount = Math.ceil(contextText.length / 4); // Assuming 4 characters per token
            // console.log(`Estimated token count: ${tokenCount}`);
        } catch (error) {
            console.error('Error in context command:', error);
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.error, "Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente mais tarde.", false)
            );
        }
    },
};
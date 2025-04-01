import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { AIService } from "../../services/AIService";
import { EMOJIS, DISCORD } from "../../utils/constants";
import { SecurityService } from "../../services/SecurityService";
import { UIService } from "../../services/UIService";
import { MessageService } from "../../services/MessageService";
import { ConversationService } from "../../services/ConversationService";
import { TextProcessingService } from "@/services/ai/TextProcessingService";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("ask")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .setDescription("Faça uma pergunta ao bot com acesso à informações externas (pesquisa web)")
        .addStringOption(option => 
            option.setName("question")
                .setDescription("A pergunta que você deseja fazer")
                .setRequired(true))
        .addChannelOption(option => 
            option.setName("context_channel")
                .setDescription("Canal opcional para usar como contexto adicional")
                .setRequired(false))
        .addIntegerOption(option =>
            option.setName("context_limit")
                .setDescription("Limite de mensagens para contexto (padrão 0 para auto-cache)")
                .setMinValue(0)
                .setMaxValue(50000)
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("ephemeral")
                .setDescription("Apenas você pode ver a resposta (padrão: false)")
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
            await interaction.deferReply({ flags: ephemeral ? MessageFlags.Ephemeral : undefined });

            const question = interaction.options.getString("question");
            let contextChannel = interaction.options.getChannel("context_channel") /* || interaction.channel; */
            let contextLimit = interaction.options.getInteger("context_limit"); // Use 0 for auto cache size
            let useCache = true; // Default to true for cache usage

            if (!contextChannel) {
                //console.warn("Context channel not provided, using current channel as default.");

                if (interaction.channel instanceof TextChannel) {
                    contextChannel = interaction.channel;
                    contextLimit = 20;
                    useCache = false; // Use cache only if context channel is provided
                }
            } else if (!contextLimit) {
                contextLimit = 0; // Default to 0 for auto cache size if no limit is provided
            }
            
            if (!question) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.info, "Por favor, forneça uma pergunta.", false));
            }

            // Process context channel if provided
            let chatContext = "";
            if (contextChannel && contextChannel instanceof TextChannel) {
                await interaction.editReply(UIService.formatStatusMessage(
                    EMOJIS.search, 
                    `Buscando contexto em \`**\`${contextChannel.name}\`**\` e informações na web sobre: "${question}"...`
                ));

                // Check channel access
                if (!contextChannel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                    return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.error, "Eu não tenho permissão para ver o canal de contexto.", false));
                }

                try {
                    // Fetch messages from context channel with limit 0 to use cache size or default
                    const messages = await MessageService.fetchMessages(contextChannel, contextLimit ?? 20, interaction, useCache);
                    
                    if (messages.length > 0) {
                        // Filter and process messages
                        const filteredMessages = MessageService.filterCommandMessages(messages, interaction);
                        const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
                        const validConversations = ConversationService.filterValidConversations(conversations, true);
                        
                        if (validConversations.length > 0) {
                            // Select relevant conversations and format as context
                            const selectedConversations = AIService.selectConversationsForContext(validConversations, question, 5000);
                            chatContext = AIService.formatConversationsAsContext(selectedConversations);
                            
                            await interaction.editReply(UIService.formatStatusMessage(
                                EMOJIS.merge, 
                                `Combinando ${selectedConversations.length} conversas relevantes com informações da web...`
                            ));
                        }
                    }
                } catch (error) {
                    console.error('Error fetching context:', error);
                    await interaction.editReply(UIService.formatStatusMessage(
                        EMOJIS.warning, 
                        `Não foi possível obter contexto do canal. Continuando apenas com pesquisa web...`
                    ));
                }
            } else {
                await interaction.editReply(UIService.formatStatusMessage(EMOJIS.search, `Procurando informações sobre: "${question}"...`));
            }
            
            // Generate response with web search and optional context
            const response = await AIService.generateWebSearchResponse(question, chatContext);
            
            // Format the response
            const messageHeader = UIService.formatStatusMessage(
                EMOJIS.complete, 
                chatContext ? `Resposta com informações da web e contexto do canal` : `Resposta com informações da web`
            );
            
            const questionContent = `**Sua pergunta:** ${question}`;
            const responseContent = `**Resposta:**\n${response}`;

            // Split message if it's too long for a single Discord message
            //const fullContent = messageHeader + "\n\n" /* + questionContent + "\n\n" */ + responseContent;
            const fullContent = messageHeader + "\n\n" /* + questionContent + "\n\n" */ + response;

            if (fullContent.length <= DISCORD.MESSAGE_LIMIT) {
                // If the message fits in a single Discord message
                await interaction.editReply({ content: fullContent });
            } else {
                // Send the header and question in the first message
                const firstMessage = messageHeader + "\n\n" + questionContent;
                await interaction.editReply({ content: firstMessage });
                
                // Split the response into chunks
                const chunks = TextProcessingService.splitLongMessage(responseContent);
                
                // Send each chunk as a follow-up message
                for (const chunk of chunks) {
                    await interaction.followUp({ 
                        content: chunk,
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined
                    });
                }
            }

        } catch (error) {
            console.error('Error in ask command:', error);
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.error, "Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente mais tarde.", false)
            );
        }
    },
};
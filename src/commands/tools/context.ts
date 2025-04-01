import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { MessageService } from "../../services/MessageService";
import { ConversationService } from "../../services/ConversationService";
import { AIService } from "../../services/AIService";
import { EMOJIS, DISCORD } from "../../utils/constants";
import { SecurityService } from "../../services/SecurityService";
import { UIService } from "../../services/UIService";

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
            // Check if user is admin
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
            const providedLimit = interaction.options.getInteger("limit");
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

            try {
                // Use 0 as the limit if not provided, so MessageService will use cache size or fallback to 1000
                const limit = providedLimit || 0;
                
                // Fetch messages using MessageService
                const messages = await MessageService.fetchMessages(channel, limit, interaction);

                //console.log(`Fetched ${messages.length} messages from channel ${channel.name}`);

                if (messages.length === 0) {
                    return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "Nenhuma mensagem encontrada no canal.", false));
                }

                // Filter messages using MessageService
                const filteredMessages = MessageService.filterCommandMessages(messages, interaction);
                
                // Group messages into conversations using ConversationService
                const conversations = ConversationService.groupMessagesByConversation(filteredMessages);
                const validConversations = ConversationService.filterValidConversations(conversations, includeBots);
                
                if (validConversations.length === 0) {
                    return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "Não foi possível extrair conversas válidas do canal.", false));
                }

                await interaction.editReply(UIService.formatStatusMessage(EMOJIS.conversation, `Processando ${validConversations.length} conversas como contexto...`));
                
                // Select relevant conversations and format them as context using AIService
                const selectedConversations = AIService.selectConversationsForContext(validConversations, prompt);
                const contextText = AIService.formatConversationsAsContext(selectedConversations);
                
                const response = await AIService.generateContextualResponse(prompt, contextText);
                
                const messageHeader = UIService.formatStatusMessage(EMOJIS.complete, `Resposta baseada no contexto`);
                // Split message if it's too long for a single Discord message
                const fullContent = messageHeader + "\n\n" + response;
                
                if (fullContent.length <= DISCORD.MESSAGE_LIMIT) {
                    // If the message fits in a single Discord message
                    await interaction.editReply({ content: fullContent });
                } else {
                    const chunks = splitLongMessage(fullContent);

                    // send the header and a chunk in the first message
                    const firstChunk = chunks.shift() || "";
                    await interaction.editReply({ 
                        content: firstChunk
                    });

                    // send the rest of the chunks as follow-up messages
                    for (const chunk of chunks) {
                        await interaction.followUp({ 
                            content: chunk,
                            flags: ephemeral ? MessageFlags.Ephemeral : undefined
                        });
                    }
                }

            } catch (error) {
                console.error('Error in context command:', error);
                throw error;
            }

        } catch (error) {
            console.error('Error in context command:', error);
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.error, "Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente mais tarde.", false)
            );
        }
    },
};

/**
 * Splits a long message into chunks that fit within Discord's message limit
 * @param message The message to split
 * @param limit The maximum length per chunk (default: Discord's message limit)
 * @returns Array of message chunks
 */
function splitLongMessage(message: string, limit: number = DISCORD.MESSAGE_LIMIT): string[] {
    const chunks: string[] = [];
    
    // If message is already within limit, return it as is
    if (message.length <= limit) {
        return [message];
    }
    
    let currentChunk = '';
    // Split by paragraphs (double newlines) first to maintain logical structure
    const paragraphs = message.split('\n\n');
    
    for (const paragraph of paragraphs) {
        // If adding this paragraph would exceed the limit, push current chunk and start a new one
        if ((currentChunk + paragraph + '\n\n').length > limit) {
            // If the paragraph itself is too long, split it further
            if (paragraph.length > limit) {
                // First push current chunk if it exists
                if (currentChunk) {
                    chunks.push(currentChunk);
                    currentChunk = '';
                }
                
                // Split long paragraph by sentences and try to keep sentences together
                const sentences = paragraph.split(/(?<=\.|\?|\!) /);
                for (const sentence of sentences) {
                    if ((currentChunk + sentence + ' ').length <= limit) {
                        currentChunk += sentence + ' ';
                    } else {
                        // If the sentence itself is too long, split by words
                        if (sentence.length > limit) {
                            if (currentChunk) {
                                chunks.push(currentChunk);
                                currentChunk = '';
                            }
                            
                            // Split by words
                            let words = sentence.split(' ');
                            for (const word of words) {
                                if ((currentChunk + word + ' ').length <= limit) {
                                    currentChunk += word + ' ';
                                } else {
                                    chunks.push(currentChunk);
                                    currentChunk = word + ' ';
                                }
                            }
                        } else {
                            chunks.push(currentChunk);
                            currentChunk = sentence + ' ';
                        }
                    }
                }
            } else {
                // Paragraph fits in a new chunk
                chunks.push(currentChunk);
                currentChunk = paragraph + '\n\n';
            }
        } else {
            // Add paragraph to current chunk
            currentChunk += paragraph + '\n\n';
        }
    }
    
    // Add the last chunk if it's not empty
    if (currentChunk) {
        chunks.push(currentChunk);
    }
    
    return chunks;
}
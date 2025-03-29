import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, EmbedBuilder, Collection, Message, ActionRowBuilder, ButtonBuilder, ButtonStyle, ComponentType } from "discord.js";
import { GoogleGenerativeAI } from "@google/generative-ai";

// Initialize the Google Generative AI
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");

interface ConversationWithContext {
    messages: Message[];
    relevanceScore: number;
}

// Emojis para melhorar a apresentação
const EMOJIS = {
    search: "🔍",
    conversation: "💬",
    page: "📄",
    relevance: "⭐",
    channel: "📌",
    time: "⏱️",
    error: "❌",
    success: "✅",
    warning: "⚠️"
};

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
                .setDescription("Número máximo de mensagens para buscar (padrão: 1000)")
                .setMinValue(1)
                .setMaxValue(20000)
                .setRequired(false)),

    async execute(interaction: ChatInputCommandInteraction) {
        try {
            await interaction.deferReply();

            const channel = interaction.options.getChannel("channel");
            const topic = interaction.options.getString("topic");
            const limit = interaction.options.getInteger("limit") || 1000;

            if (!channel || !topic) {
                return await interaction.editReply(`${EMOJIS.error} Por favor, forneça um canal e um tópico para buscar.`);
            }

            if (!(channel instanceof TextChannel)) {
                return await interaction.editReply(`${EMOJIS.error} O canal deve ser um canal de texto.`);
            }

            // Check if the bot has permission to view the channel
            if (!channel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                return await interaction.editReply(`${EMOJIS.error} Eu não tenho permissão para ver esse canal.`);
            }

            await interaction.editReply(`${EMOJIS.search} Buscando mensagens sobre **"${topic}"** em ${channel}...`);

            // Fetch messages from the channel
            const messages = await fetchMessages(channel, limit, interaction);

            //console.log(`Fetched ${messages.length} messages from ${channel.name}`);

            // print content of the first and last message
            if (messages.length > 0) {
                //console.log(`First message: ${messages[0].content}`);
                //console.log(`Last message: ${messages[messages.length - 1].content}`);
            }

            if (messages.length === 0) {
                return await interaction.editReply(`${EMOJIS.warning} Nenhuma mensagem encontrada no canal.`);
            }

            // Filtrar mensagens para excluir a mensagem do próprio comando de busca
            const filteredMessages = messages.filter(msg => {
                // Excluir mensagens do próprio autor do comando nos últimos segundos
                if (msg.author.id === interaction.user.id) {
                    // Se a mensagem foi enviada nos últimos 10 segundos, provavelmente é o comando atual
                    const timeDiff = interaction.createdTimestamp - msg.createdTimestamp;
                    if (Math.abs(timeDiff) < 10000) { // 10 segundos
                        return false;
                    }
                }
                return true;
            });

            // Process the messages with Google's Generative AI
            const relevantConversations = await findRelevantConversations(filteredMessages, topic, interaction);

            if (relevantConversations.length === 0) {
                return await interaction.editReply(`${EMOJIS.warning} Nenhuma mensagem relacionada a **"${topic}"** encontrada em ${channel}.`);
            }

            // Display the results with pagination
            await displayPaginatedResults(interaction, relevantConversations, topic, channel.name);

        } catch (error) {
            console.error('Erro no comando de busca:', error);
            await interaction.editReply(`${EMOJIS.error} Ocorreu um erro durante a busca. Por favor, tente novamente mais tarde.`);
        }
    },
};

async function fetchMessages(channel: TextChannel, limit: number, interaction: ChatInputCommandInteraction): Promise<Message[]> {
    const messages: Message[] = [];
    let lastId: string | undefined;
    const batchSize = 100;
    let consecutiveEmptyFetches = 0;
    const maxEmptyFetches = 3;
    let totalFetches = 0;
    const maxFetchAttempts = Math.ceil(limit / batchSize) + maxEmptyFetches;

    while (messages.length < limit && consecutiveEmptyFetches < maxEmptyFetches && totalFetches < maxFetchAttempts) {
        try {
            const options: { limit: number; before?: string } = { 
                limit: Math.min(batchSize, limit - messages.length)
            };
            if (lastId) {
                options.before = lastId;
            }

            const fetchedMessages = await channel.messages.fetch(options);

            totalFetches++;

            if (!fetchedMessages || fetchedMessages.size === 0) {
                consecutiveEmptyFetches++;
                if (consecutiveEmptyFetches >= maxEmptyFetches) {
                    console.log('Stopping fetch: No more messages found after multiple attempts');
                    break;
                }
            } else {
                consecutiveEmptyFetches = 0;
                const fetchedArray = [...fetchedMessages.values()];
                messages.push(...fetchedArray);
                const lastMessage = fetchedArray[fetchedArray.length - 1];
                lastId = lastMessage?.id;

                if (messages.length % 1000 === 0) {
                    await interaction.editReply(
                        `${EMOJIS.search} Buscando mensagens... (${messages.length}/${limit} mensagens encontradas)`
                    );
                }

                if (totalFetches % 5 === 0) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
        } catch (error) {
            console.error('Error fetching messages:', error);
            if (error && typeof error === 'object' && 'code' in error && error.code === 50001) {
                throw new Error('Não tenho permissão para ler mensagens neste canal.');
            }
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }

    return messages;
}

async function findRelevantConversations(messages: Message[], topic: string, interaction: ChatInputCommandInteraction): Promise<ConversationWithContext[]> {
    try {
        const conversations = groupMessagesByConversation(messages);
        const validConversations = conversations.filter(conversation => {
            if (conversation.length === 1 && conversation[0].content.length < 10) return false;
            if (conversation.every(msg => msg.author.bot)) return false;
            return true;
        });

        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
        const relevantConversations: ConversationWithContext[] = [];

        // Dynamically adjust batch size based on number of conversations
        // Gemini has a 1M token context window, so we can process more at once
        // Assuming average conversation is 200 tokens, we can safely process about 4000 conversations
        // But let's be conservative and use smaller batches for better reliability
        const calculateBatchSize = (totalConversations: number) => {
            if (totalConversations <= 10) return 10;
            if (totalConversations <= 50) return 25;
            if (totalConversations <= 100) return 50;
            if (totalConversations <= 500) return 100;
            return 200; // Max batch size for very large sets
        };

        const batchSize = calculateBatchSize(validConversations.length);
        console.log(`Processing ${validConversations.length} conversations in batches of ${batchSize}`);

        // Process conversations in batches
        for (let i = 0; i < validConversations.length; i += batchSize) {
            const batch = validConversations.slice(i, i + batchSize);
            
            // Create text representation of all conversations in the batch
            const batchConversationsText = batch.map((conversation, index) => {
                const conversationText = conversation
                    .map(msg => `${msg.author.username}: ${msg.content}`)
                    .join("\n");
                return `[Conversation ${i + index + 1}]\n${conversationText}\n`;
            }).join("\n---\n");

            if (!batchConversationsText.trim()) continue;

            // Update progress
            if (i % (batchSize * 2) === 0) {
                await interaction.editReply(
                    `${EMOJIS.search} Analisando conversas... (${i}/${validConversations.length} processadas)`
                );
            }

            const prompt = `
            Analyze the following ${batch.length} conversations and determine if they are related to the topic: "${topic}".
            For each conversation, determine:
            1. If it's relevant (YES/NO)
            2. A relevance score (0-10)
            
            Return the results in exactly this format, one line per conversation:
            CONV1: YES/NO: SCORE
            CONV2: YES/NO: SCORE
            etc.
            
            Conversations:
            ${batchConversationsText}
            `;

            const result = await model.generateContent(prompt);
            const resultText = result.response.text().trim();
            
            // Parse results
            const results = resultText.split('\n');
            
            batch.forEach((conversation, index) => {
                const resultLine = results[index];
                if (!resultLine) return;
                
                const isRelevant = resultLine.includes("YES");
                const scoreMatch = resultLine.match(/:\s*(\d+)/);
                const relevanceScore = scoreMatch ? parseInt(scoreMatch[1], 10) : 5;

                if (isRelevant) {
                    relevantConversations.push({
                        messages: conversation,
                        relevanceScore
                    });
                }
            });
        }

        // Sort by relevance score (highest first)
        return relevantConversations.sort((a, b) => b.relevanceScore - a.relevanceScore);

    } catch (error) {
        console.error('Erro ao analisar mensagens com IA:', error);
        // Fallback: Basic keyword matching
        const fallbackConversations: ConversationWithContext[] = [];
        const conversations = groupMessagesByConversation(messages);

        for (const conversation of conversations) {
            if (conversation.some(msg => msg.content.toLowerCase().includes(topic.toLowerCase()))) {
                fallbackConversations.push({
                    messages: conversation,
                    relevanceScore: 5 // Default score for fallback matches
                });
            }
        }

        return fallbackConversations;
    }
}

function groupMessagesByConversation(messages: Message[]): Message[][] {
    const conversations: Message[][] = [];
    let currentConversation: Message[] = [];
    const MAX_CONVERSATION_GAP = 5 * 60 * 1000;   // 5 minutes for regular messages
    const SAME_AUTHOR_GAP = 8 * 60 * 1000;        // 8 minutes for same author
    const REPLY_GAP = 10 * 60 * 1000;             // 10 minutes for direct replies

    // Sort messages by timestamp (oldest first)
    const sortedMessages = [...messages].sort((a, b) => a.createdTimestamp - b.createdTimestamp);

    const isRelatedToLastMessage = (message: Message): boolean => {
        if (!currentConversation.length) return true;
        
        const lastMessage = currentConversation[currentConversation.length - 1];
        const timeDiff = message.createdTimestamp - lastMessage.createdTimestamp;
        
        // If it's a direct reply to any message in the current conversation
        if (message.reference?.messageId && 
            currentConversation.some(msg => msg.id === message.reference?.messageId)) {
            return timeDiff <= REPLY_GAP;
        }

        // If it's the same author
        if (message.author.id === lastMessage.author.id) {
            return timeDiff <= SAME_AUTHOR_GAP;
        }

        // For any other message
        return timeDiff <= MAX_CONVERSATION_GAP;
    };

    for (const message of sortedMessages) {
        // Skip empty messages
        if (!message.content.trim()) continue;

        if (!isRelatedToLastMessage(message)) {
            if (currentConversation.length > 0) {
                conversations.push([...currentConversation]);
                currentConversation = [];
            }
        }

        currentConversation.push(message);
    }

    if (currentConversation.length > 0) {
        conversations.push(currentConversation);
    }

    return conversations;
}

async function displayPaginatedResults(
    interaction: ChatInputCommandInteraction,
    conversations: ConversationWithContext[],
    topic: string,
    channelName: string
) {
    let currentConvIndex = 0;
    let currentMsgIndex = 0;

    // Function to create embed for the current page
    const createEmbed = () => {
        const conversation = conversations[currentConvIndex];
        const totalConversations = conversations.length;

        // Calculate total messages in the current conversation and how many pages we'll have
        const currentConvMessages = conversation.messages;
        const totalMessagesInConv = currentConvMessages.length;
        const maxMessagesPerPage = 5;

        // Obter a primeira mensagem da conversa para criar um link
        const firstMsg = currentConvMessages[0];
        const msgLink = firstMsg ? `https://discord.com/channels/${firstMsg.guild?.id}/${firstMsg.channel.id}/${firstMsg.id}` : '';

        // Calculate start and end index for messages to show on this page
        const startIdx = currentMsgIndex;
        const endIdx = Math.min(startIdx + maxMessagesPerPage, totalMessagesInConv);

        // Create a header for the embed
        const embed = new EmbedBuilder()
            .setTitle(`${EMOJIS.search} Resultados para "${topic}"`)
            .setDescription(
                `${EMOJIS.conversation} **Conversa ${currentConvIndex + 1} de ${totalConversations}**\n` +
                `${EMOJIS.relevance} Relevância: ${conversation.relevanceScore}/10\n` +
                `${EMOJIS.channel} Canal: ${channelName}\n` +
                (msgLink ? `[➡️ Ir para a conversa](${msgLink})` : '')
            )
            .setColor(0x7289da)
            .setFooter({
                text: `${EMOJIS.page} Página ${Math.floor(startIdx / maxMessagesPerPage) + 1} de ${Math.ceil(totalMessagesInConv / maxMessagesPerPage)}`
            })
            .setTimestamp();

        // Agrupar mensagens do mesmo autor
        const messages = currentConvMessages.slice(startIdx, endIdx);

        // Agrupar mensagens por autor
        const messagesByAuthor = new Map<string, { author: string, content: string[], timestamp: number }>();

        for (const msg of messages) {
            const authorId = msg.author.id;
            const content = msg.content || "*Sem conteúdo*";

            if (!messagesByAuthor.has(authorId)) {
                messagesByAuthor.set(authorId, {
                    author: msg.author.username,
                    content: [content],
                    timestamp: msg.createdTimestamp
                });
            } else {
                messagesByAuthor.get(authorId)!.content.push(content);
            }
        }

        // Ordenar por timestamp
        const sortedAuthors = Array.from(messagesByAuthor.values())
            .sort((a, b) => a.timestamp - b.timestamp);

        // Adicionar campos para cada autor
        for (const authorData of sortedAuthors) {
            const date = new Date(authorData.timestamp).toLocaleDateString('pt-BR', {
                day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
            });

            // Combinar todas as mensagens do mesmo autor
            let combinedContent = '';

            for (const content of authorData.content) {
                // Se adicionar esta mensagem exceder o limite, truncar
                const potentialContent = combinedContent + (combinedContent ? '\n' : '') + content;

                if (potentialContent.length > 1000) {
                    // Se o conteúdo combinado ficar muito grande, adicionar o campo atual
                    // e começar um novo campo para o mesmo autor
                    embed.addFields({
                        name: `${authorData.author} • ${date}`,
                        value: combinedContent || "*Sem conteúdo*"
                    });

                    combinedContent = content;
                } else {
                    combinedContent = potentialContent;
                }
            }

            // Adicionar o último (ou único) campo para este autor
            if (combinedContent) {
                embed.addFields({
                    name: `${authorData.author} • ${date}`,
                    value: combinedContent
                });
            }
        }

        return embed;
    };

    // Create the initial embed and button row
    const embed = createEmbed();

    // Create the navigation buttons
    const getButtonRow = () => {
        const row = new ActionRowBuilder<ButtonBuilder>()
            .addComponents(
                new ButtonBuilder()
                    .setCustomId('prev_conv')
                    .setLabel('◀️ Conversa Anterior')
                    .setStyle(ButtonStyle.Primary)
                    .setDisabled(currentConvIndex === 0),
                new ButtonBuilder()
                    .setCustomId('prev_page')
                    .setLabel('◀️ Página Anterior')
                    .setStyle(ButtonStyle.Secondary)
                    .setDisabled(currentMsgIndex === 0),
                new ButtonBuilder()
                    .setCustomId('next_page')
                    .setLabel('Próxima Página ▶️')
                    .setStyle(ButtonStyle.Secondary)
                    .setDisabled(currentMsgIndex + 5 >= conversations[currentConvIndex].messages.length),
                new ButtonBuilder()
                    .setCustomId('next_conv')
                    .setLabel('Próxima Conversa ▶️')
                    .setStyle(ButtonStyle.Primary)
                    .setDisabled(currentConvIndex === conversations.length - 1)
            );
        return row;
    };

    // Send the initial message with buttons
    const message = await interaction.editReply({
        content: `${EMOJIS.success} Busca completada! Encontrei ${conversations.length} conversas sobre **"${topic}"**.`,
        embeds: [embed],
        components: [getButtonRow()]
    });

    // Create a collector for button interactions
    const collector = message.createMessageComponentCollector({
        componentType: ComponentType.Button,
        time: 300000 // 5 minutes timeout
    });

    collector.on('collect', async (i) => {
        // Only respond to the user who initiated the command
        if (i.user.id !== interaction.user.id) {
            await i.reply({ content: `${EMOJIS.error} Apenas o autor do comando pode interagir com esses botões.`, ephemeral: true });
            return;
        }

        switch (i.customId) {
            case 'prev_conv':
                currentConvIndex = Math.max(0, currentConvIndex - 1);
                currentMsgIndex = 0;
                break;
            case 'next_conv':
                currentConvIndex = Math.min(conversations.length - 1, currentConvIndex + 1);
                currentMsgIndex = 0;
                break;
            case 'prev_page':
                currentMsgIndex = Math.max(0, currentMsgIndex - 5);
                break;
            case 'next_page':
                currentMsgIndex = Math.min(conversations[currentConvIndex].messages.length - 1, currentMsgIndex + 5);
                break;
        }

        // Update the embed and buttons
        await i.update({
            embeds: [createEmbed()],
            components: [getButtonRow()]
        });
    });

    collector.on('end', async () => {
        // Remove buttons when the collector times out
        await interaction.editReply({
            content: `${EMOJIS.warning} Esta sessão de busca expirou.`,
            embeds: [createEmbed().setFooter({ text: 'Esta sessão de busca expirou. Execute o comando novamente para uma nova busca.' })],
            components: []
        }).catch(console.error);
    });
}
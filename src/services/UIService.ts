import { 
    ChatInputCommandInteraction, 
    EmbedBuilder, 
    ActionRowBuilder, 
    ButtonBuilder, 
    ButtonStyle, 
    ComponentType,
    Message
} from "discord.js";
import { ConversationWithContext, MessageGroup } from "../types/conversation";
import { EMOJIS, ADMIN_IDS } from "../utils/constants";

export class UIService {
    private static readonly MAX_MESSAGES_PER_PAGE = 5;
    private static readonly COLLECTOR_TIMEOUT = 300000; // 5 minutes
    private static readonly MAX_FIELD_VALUE_LENGTH = 1024;
    private static readonly MAX_FIELD_NAME_LENGTH = 256;

    public static async displaySearchResults(
        interaction: ChatInputCommandInteraction,
        conversations: ConversationWithContext[],
        topic: string,
        channelName: string,
        ephemeral: boolean = false
    ): Promise<void> {
        const currentState = {
            currentConvIndex: 0,
            currentMsgIndex: 0
        };

        const embed = this.createEmbed(conversations[0], currentState.currentMsgIndex, topic, channelName, conversations.length, currentState.currentConvIndex);
        const row = this.createButtonRow(currentState.currentConvIndex, currentState.currentMsgIndex, conversations);

        const message = await interaction.editReply({
            content: `${EMOJIS.success} Busca completada! Encontrei ${conversations.length} conversas sobre **"${topic}"**.`,
            embeds: [embed],
            components: [row]
        });

        const collector = message.createMessageComponentCollector({
            componentType: ComponentType.Button,
            time: this.COLLECTOR_TIMEOUT
        });

        this.handleCollector(collector, interaction, conversations, topic, channelName, currentState, ephemeral);
    }

    private static createEmbed(
        conversation: ConversationWithContext,
        startIndex: number,
        topic: string,
        channelName: string,
        totalConversations: number,
        currentConvIndex: number
    ): EmbedBuilder {
        const messages = conversation.messages;
        const totalMessages = messages.length;
        const firstMsg = messages[0];
        const msgLink = firstMsg ? 
            `https://discord.com/channels/${firstMsg.guild?.id}/${firstMsg.channel.id}/${firstMsg.id}` : '';

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
                text: `${EMOJIS.page} Página ${Math.floor(startIndex / this.MAX_MESSAGES_PER_PAGE) + 1} de ${Math.ceil(totalMessages / this.MAX_MESSAGES_PER_PAGE)}`
            })
            .setTimestamp();

        const messageGroups = this.groupMessagesByAuthor(
            messages.slice(startIndex, startIndex + this.MAX_MESSAGES_PER_PAGE)
        );

        for (const group of messageGroups) {
            const date = new Date(group.timestamp).toLocaleDateString('pt-BR', {
                day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
            });

            const baseFieldName = `${group.author} • ${date}`;
            const chunks: string[] = [];
            let currentChunk = '';

            for (const content of group.content) {
                if (!content.trim()) continue;

                // Split long messages into smaller parts if needed
                const contentParts = this.splitContentIntoChunks(content);
                
                for (const part of contentParts) {
                    if ((currentChunk + '\n' + part).length <= this.MAX_FIELD_VALUE_LENGTH) {
                        currentChunk = currentChunk ? currentChunk + '\n' + part : part;
                    } else {
                        if (currentChunk) chunks.push(currentChunk);
                        currentChunk = part;
                    }
                }
            }

            if (currentChunk) {
                chunks.push(currentChunk);
            }

            // Add fields with proper length validation
            chunks.forEach((chunk, index) => {
                const fieldName = chunks.length > 1 
                    ? `${baseFieldName} (${index + 1}/${chunks.length})`
                    : baseFieldName;

                // Ensure field name isn't too long
                const truncatedName = fieldName.length > this.MAX_FIELD_NAME_LENGTH
                    ? fieldName.substring(0, this.MAX_FIELD_NAME_LENGTH - 3) + '...'
                    : fieldName;

                embed.addFields({
                    name: truncatedName,
                    value: chunk.substring(0, this.MAX_FIELD_VALUE_LENGTH)
                });
            });
        }

        return embed;
    }

    private static splitContentIntoChunks(content: string): string[] {
        if (content.length <= this.MAX_FIELD_VALUE_LENGTH) {
            return [content];
        }

        const chunks: string[] = [];
        let currentIndex = 0;

        while (currentIndex < content.length) {
            // Try to split at the last space within the limit
            let endIndex = currentIndex + this.MAX_FIELD_VALUE_LENGTH;
            if (endIndex < content.length) {
                const lastSpace = content.lastIndexOf(' ', endIndex);
                if (lastSpace > currentIndex) {
                    endIndex = lastSpace;
                }
            }

            chunks.push(content.slice(currentIndex, endIndex).trim());
            currentIndex = endIndex;
        }

        return chunks;
    }

    private static createButtonRow(
        currentConvIndex: number,
        currentMsgIndex: number,
        conversations: ConversationWithContext[]
    ): ActionRowBuilder<ButtonBuilder> {
        return new ActionRowBuilder<ButtonBuilder>()
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
                    .setDisabled(currentMsgIndex + this.MAX_MESSAGES_PER_PAGE >= conversations[currentConvIndex].messages.length),
                new ButtonBuilder()
                    .setCustomId('next_conv')
                    .setLabel('Próxima Conversa ▶️')
                    .setStyle(ButtonStyle.Primary)
                    .setDisabled(currentConvIndex === conversations.length - 1)
            );
    }

    private static groupMessagesByAuthor(messages: Message[]): MessageGroup[] {
        const groups = new Map<string, MessageGroup>();

        for (const msg of messages) {
            const authorId = msg.author.id;
            const content = msg.content || "*Sem conteúdo*";

            if (!groups.has(authorId)) {
                groups.set(authorId, {
                    author: msg.author.username,
                    content: [content],
                    timestamp: msg.createdTimestamp
                });
            } else {
                groups.get(authorId)!.content.push(content);
            }
        }

        return Array.from(groups.values())
            .sort((a, b) => a.timestamp - b.timestamp);
    }

    private static handleCollector(
        collector: any,
        interaction: ChatInputCommandInteraction,
        conversations: ConversationWithContext[],
        topic: string,
        channelName: string,
        state: { currentConvIndex: number; currentMsgIndex: number },
        ephemeral: boolean = false
    ): void {
        collector.on('collect', async (i: any) => {
            if (i.user.id !== interaction.user.id && !ADMIN_IDS.includes(i.user.id)) {
                await i.reply({ 
                    content: `${EMOJIS.error} Apenas o autor do comando e administradores podem interagir com esses botões.`, 
                    ephemeral: true 
                });
                return;
            }

            switch (i.customId) {
                case 'prev_conv':
                    state.currentConvIndex = Math.max(0, state.currentConvIndex - 1);
                    state.currentMsgIndex = 0;
                    break;
                case 'next_conv':
                    state.currentConvIndex = Math.min(conversations.length - 1, state.currentConvIndex + 1);
                    state.currentMsgIndex = 0;
                    break;
                case 'prev_page':
                    state.currentMsgIndex = Math.max(0, state.currentMsgIndex - this.MAX_MESSAGES_PER_PAGE);
                    break;
                case 'next_page':
                    state.currentMsgIndex = Math.min(
                        conversations[state.currentConvIndex].messages.length - 1, 
                        state.currentMsgIndex + this.MAX_MESSAGES_PER_PAGE
                    );
                    break;
            }

            const embed = this.createEmbed(
                conversations[state.currentConvIndex],
                state.currentMsgIndex,
                topic,
                channelName,
                conversations.length,
                state.currentConvIndex
            );

            const row = this.createButtonRow(state.currentConvIndex, state.currentMsgIndex, conversations);

            await i.update({ embeds: [embed], components: [row] });
        });

        collector.on('end', async () => {
            const embed = this.createEmbed(
                conversations[state.currentConvIndex],
                state.currentMsgIndex,
                topic,
                channelName,
                conversations.length,
                state.currentConvIndex
            ).setFooter({ 
                text: 'Esta sessão de busca expirou. Execute o comando novamente para uma nova busca.' 
            });

            await interaction.editReply({
                content: `${EMOJIS.warning} Esta sessão de busca expirou.`,
                embeds: [embed],
                components: []
            }).catch(console.error);
        });
    }
}
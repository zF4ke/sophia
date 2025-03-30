import { 
    ChatInputCommandInteraction, 
    EmbedBuilder, 
    ActionRowBuilder, 
    ButtonBuilder,
    ButtonStyle,
    ComponentType,
    Message,
    MessageFlags
} from "discord.js";
import { MessageGroup } from "../types/conversation";
import { EMOJIS } from "../utils/constants";
import { SecurityService } from "./SecurityService";

export interface NavigationButton {
    customId: string;
    label: string;
    style: 'Primary' | 'Secondary' | 'Success' | 'Danger';
    disabled: boolean;
}

export interface NavigationState {
    currentConvIndex: number;
    currentMsgIndex: number;
}

export class UIService {
    protected static readonly MAX_ITEMS_PER_PAGE = 5;
    protected static readonly COLLECTOR_TIMEOUT = 300000; // 5 minutes
    protected static readonly MAX_FIELD_VALUE_LENGTH = 1024;
    protected static readonly MAX_FIELD_NAME_LENGTH = 256;
    protected static readonly DEFAULT_COLOR = 0x7289da;

    /**
     * Creates a formatted status message with backticks and emoji
     * @param emoji The emoji character to use
     * @param message The status message text
     * @param useBackticks Whether to format with backticks or return plain text
     * @returns Formatted status message
     */
    public static formatStatusMessage(emoji: string, message: string, useBackticks = true): string {
        return useBackticks ? `\`${emoji} ${message}\`` : `${emoji} ${message}`;
    }

    protected static createNavigationRow(buttons: NavigationButton[]): ActionRowBuilder<ButtonBuilder> {
        return new ActionRowBuilder<ButtonBuilder>()
            .addComponents(
                ...buttons.map(btn => 
                    new ButtonBuilder()
                        .setCustomId(btn.customId)
                        .setLabel(btn.label)
                        .setStyle(ButtonStyle[btn.style])
                        .setDisabled(btn.disabled)
                )
            );
    }

    protected static addMessageGroupFields(embed: EmbedBuilder, messages: Message[]): void {
        const messageGroups = this.groupMessagesByAuthor(messages);

        for (const group of messageGroups) {
            const date = new Date(group.timestamp).toLocaleDateString('pt-BR', {
                day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
            });

            const baseFieldName = `${group.author} • ${date}`;
            const chunks: string[] = [];
            let currentChunk = '';

            for (const content of group.content) {
                if (!content.trim()) continue;
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

            if (currentChunk) chunks.push(currentChunk);

            chunks.forEach((chunk, index) => {
                const fieldName = chunks.length > 1 
                    ? `${baseFieldName} (${index + 1}/${chunks.length})`
                    : baseFieldName;

                const truncatedName = fieldName.length > this.MAX_FIELD_NAME_LENGTH
                    ? fieldName.substring(0, this.MAX_FIELD_NAME_LENGTH - 3) + '...'
                    : fieldName;

                embed.addFields({
                    name: truncatedName,
                    value: chunk.substring(0, this.MAX_FIELD_VALUE_LENGTH)
                });
            });
        }
    }

    protected static splitContentIntoChunks(content: string): string[] {
        if (content.length <= this.MAX_FIELD_VALUE_LENGTH) {
            return [content];
        }

        const chunks: string[] = [];
        let currentIndex = 0;

        while (currentIndex < content.length) {
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

    protected static groupMessagesByAuthor(messages: Message[]): MessageGroup[] {
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

    protected static async setupInteractionCollector<T>(
        message: Message,
        interaction: ChatInputCommandInteraction,
        items: T[],
        state: NavigationState,
        createEmbed: (state: NavigationState) => EmbedBuilder,
        createRow: (state: NavigationState) => ActionRowBuilder<ButtonBuilder>,
        ephemeral: boolean = false
    ): Promise<void> {
        const collector = message.createMessageComponentCollector({
            componentType: ComponentType.Button,
            time: this.COLLECTOR_TIMEOUT
        });

        collector.on('collect', async (i: any) => {
            if (i.user.id !== interaction.user.id && !SecurityService.isAdmin(i.user.id)) {
                await i.reply({ 
                    content: `${EMOJIS.error} Apenas o autor do comando e administradores podem interagir com esses botões.`, 
                    flags: MessageFlags.Ephemeral
                });
                return;
            }

            let updated = false;
            switch (i.customId) {
                case 'prev_conv':
                    if (state.currentConvIndex > 0) {
                        state.currentConvIndex--;
                        state.currentMsgIndex = 0;
                        updated = true;
                    }
                    break;
                case 'next_conv':
                    if (state.currentConvIndex < items.length - 1) {
                        state.currentConvIndex++;
                        state.currentMsgIndex = 0;
                        updated = true;
                    }
                    break;
                case 'prev_page':
                    if (state.currentMsgIndex > 0) {
                        state.currentMsgIndex = Math.max(0, state.currentMsgIndex - this.MAX_ITEMS_PER_PAGE);
                        updated = true;
                    }
                    break;
                case 'next_page':
                    const maxIndex = (items[state.currentConvIndex] as any).messages?.length ?? items.length;
                    if (state.currentMsgIndex + this.MAX_ITEMS_PER_PAGE < maxIndex) {
                        state.currentMsgIndex += this.MAX_ITEMS_PER_PAGE;
                        updated = true;
                    }
                    break;
            }

            if (updated) {
                try {
                    const embed = createEmbed(state);
                    const row = createRow(state);
                    await i.update({ embeds: [embed], components: [row] });
                } catch (error: any) {
                    if (error.code === 10008) { // Unknown Message error
                        // Silently fail if the message was deleted or expired
                        collector.stop('messageDeleted');
                        return;
                    }
                    throw error;
                }
            }
        });

        collector.on('end', async (_, reason) => {
            if (reason === 'messageDeleted') return;
            
            try {
                const embed = createEmbed(state)
                    .setFooter({ 
                        text: 'Esta sessão expirou. Execute o comando novamente para uma nova sessão.' 
                    });

                await interaction.editReply({
                    embeds: [embed],
                    components: []
                });
            } catch (error: any) {
                if (error.code !== 10008) { // Only log if it's not an Unknown Message error
                    console.error('Error updating expired interaction:', error);
                }
            }
        });
    }
}
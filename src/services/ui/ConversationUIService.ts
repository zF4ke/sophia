import { 
    ChatInputCommandInteraction, 
    EmbedBuilder, 
    ActionRowBuilder, 
    ButtonBuilder,
    Message
} from "discord.js";
import { ConversationWithContext } from "../../types/conversation";
import { EMOJIS } from "../../utils/constants";
import { UIService, NavigationButton, NavigationState } from "../UIService";

export class ConversationUIService extends UIService {
    public static async displayConversations(
        interaction: ChatInputCommandInteraction,
        conversations: ConversationWithContext[],
        topic: string,
        channelName: string,
        ephemeral: boolean = false
    ): Promise<void> {
        const currentState: NavigationState = {
            currentConvIndex: 0,
            currentMsgIndex: 0
        };

        const createEmbed = (state: NavigationState): EmbedBuilder => {
            const conversation = conversations[state.currentConvIndex];
            const messages = conversation.messages;
            const totalMessages = messages.length;
            const firstMsg = messages[0];
            const msgLink = firstMsg ? 
                `https://discord.com/channels/${firstMsg.guild?.id}/${firstMsg.channel.id}/${firstMsg.id}` : '';

            const embed = new EmbedBuilder()
                .setTitle(`${EMOJIS.search} Conversa sobre "${topic}"`)
                .setDescription(
                    `${EMOJIS.conversation} **Conversa ${state.currentConvIndex + 1} de ${conversations.length}**\n` +
                    `${EMOJIS.relevance} Relevância: ${conversation.relevanceScore}/10\n` +
                    `${EMOJIS.channel} Canal: ${channelName}\n` +
                    (msgLink ? `[➡️ Ir para a conversa](${msgLink})` : '')
                )
                .setColor(this.DEFAULT_COLOR)
                .setFooter({
                    text: `${EMOJIS.page} Página ${Math.floor(state.currentMsgIndex / this.MAX_ITEMS_PER_PAGE) + 1} de ${Math.ceil(totalMessages / this.MAX_ITEMS_PER_PAGE)}`
                })
                .setTimestamp();

            this.addMessageGroupFields(embed, messages.slice(state.currentMsgIndex, state.currentMsgIndex + this.MAX_ITEMS_PER_PAGE));
            return embed;
        };

        const createRow = (state: NavigationState): ActionRowBuilder<ButtonBuilder> => {
            const buttons: NavigationButton[] = [
                {
                    customId: 'prev_conv',
                    label: '◀️ Conversa Anterior',
                    style: 'Primary',
                    disabled: state.currentConvIndex === 0
                },
                {
                    customId: 'prev_page',
                    label: '◀️ Página Anterior',
                    style: 'Secondary',
                    disabled: state.currentMsgIndex === 0
                },
                {
                    customId: 'next_page',
                    label: 'Próxima Página ▶️',
                    style: 'Secondary',
                    disabled: state.currentMsgIndex + this.MAX_ITEMS_PER_PAGE >= conversations[state.currentConvIndex].messages.length
                },
                {
                    customId: 'next_conv',
                    label: 'Próxima Conversa ▶️',
                    style: 'Primary',
                    disabled: state.currentConvIndex === conversations.length - 1
                }
            ];

            return this.createNavigationRow(buttons);
        };

        const embed = createEmbed(currentState);
        const row = createRow(currentState);

        const message = await interaction.editReply({
            content: `${EMOJIS.success} Mostrando ${conversations.length} ${conversations.length === 1 ? 'conversa' : 'conversas'} sobre **"${topic}"**.`,
            embeds: [embed],
            components: [row]
        });

        await this.setupInteractionCollector(
            message,
            interaction,
            conversations,
            currentState,
            createEmbed,
            createRow,
            ephemeral
        );
    }
}
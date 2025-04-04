import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { MessageService } from "../../services/MessageService";
import { EMOJIS } from "../../utils/constants";
import { UIService } from "../../services/UIService";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("getmessage")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .setDescription("Obter uma mensagem específica por posição em um canal")
        .addChannelOption(option =>
            option.setName("channel")
                .setDescription("O canal onde buscar a mensagem")
                .setRequired(true))
        .addIntegerOption(option =>
            option.setName("number")
                .setDescription("Número da mensagem (Máximo: 20000)")
                .setMinValue(1)
                .setMaxValue(20000)
                .setRequired(true))
        .addBooleanOption(option =>
            option.setName("ephemeral")
                .setDescription("Apenas você pode ver o resultado (padrão: false)")
                .setRequired(false)),

    async execute(interaction: ChatInputCommandInteraction) {
        try {
            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
            
            try {
                await interaction.deferReply({ flags: ephemeral ? MessageFlags.Ephemeral : undefined });
            } catch (error) {
                console.error('Failed to defer reply:', error);
                return;
            }

            const channel = interaction.options.getChannel("channel");
            const messageNumber = interaction.options.getInteger("number");

            if (!channel || messageNumber === null) {
                await safeReply(interaction, UIService.formatStatusMessage(EMOJIS.info, "Por favor, forneça um canal e um número de mensagem.", false));
                return;
            }

            if (!(channel instanceof TextChannel)) {
                await safeReply(interaction, UIService.formatStatusMessage(EMOJIS.warning, "O canal deve ser um canal de texto.", false));
                return;
            }

            if (!channel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                await safeReply(interaction, UIService.formatStatusMessage(EMOJIS.error, "Eu não tenho permissão para ver esse canal.", false));
                return;
            }

            await safeReply(interaction, UIService.formatStatusMessage(EMOJIS.loading, `Buscando a mensagem ${messageNumber} em \`**\`${channel.name}\`**\` ...`));

            try {
                // Fetch messages from start
                const messages = await MessageService.fetchMessages(channel, messageNumber, interaction);

                // Get the last message (which will be the nth message)
                const targetMessage = messages.length > 0 ? messages[0] : null;

                if (!targetMessage) {
                    await safeReply(interaction,
                        UIService.formatStatusMessage(EMOJIS.warning, `Não foi possível encontrar a mensagem ${messageNumber} em ${channel.toString()}.`, false)
                    );
                    return;
                }

                // Generate message link
                const messageLink = `https://discord.com/channels/${targetMessage.guild?.id}/${targetMessage.channel.id}/${targetMessage.id}`;
                
                // Format message content preview (truncate if needed)
                const contentPreview = targetMessage.content.length > 200 
                    ? targetMessage.content.substring(0, 200) + "..." 
                    : targetMessage.content || "*Sem conteúdo textual*";

                // Format response with message details
                const response = `${EMOJIS.found} **Mensagem ${messageNumber} em ${channel.toString()}:**

**Autor:** ${targetMessage.author.username}
**Data:** ${targetMessage.createdAt.toLocaleDateString('pt-BR')} às ${targetMessage.createdAt.toLocaleTimeString('pt-BR')}
**Conteúdo:** 
${contentPreview}

**Link:** [Clique para ir à mensagem](${messageLink})`;

                await safeReply(interaction, response);

            } catch (error) {
                console.error('Error fetching message:', error);
                throw error; // Re-throw to be caught by outer try-catch
            }

        } catch (error) {
            console.error('Error in getmessage command:', error);
            await safeReply(interaction, 
                UIService.formatStatusMessage(EMOJIS.error, "Ocorreu um erro durante a busca da mensagem. O canal pode ter muitas mensagens ou a mensagem solicitada não existe.", false)
            );
        }
    },
};

// Add this helper function at the end of the file
async function safeReply(interaction: ChatInputCommandInteraction, content: string) {
    try {
        if (interaction.replied) {
            await interaction.editReply(content);
        } else if (interaction.deferred) {
            await interaction.editReply(content);
        } else {
            await interaction.reply(content);
        }
    } catch (error) {
        console.error('Failed to reply to interaction:', error);
    }
}
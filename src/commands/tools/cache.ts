import { SlashCommandBuilder, ChatInputCommandInteraction, TextChannel, PermissionFlagsBits, MessageFlags } from "discord.js";
import { MessageService } from "../../services/MessageService";
import { EMOJIS } from "../../utils/constants";
import { SecurityService } from "../../services/SecurityService";
import { UIService } from "../../services/UIService";
import { FileSystemService } from "../../services/FileSystemService";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("cache")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .setDescription("Exibe informações sobre o cache de mensagens para um canal")
        .addChannelOption(option => 
            option.setName("channel")
                .setDescription("O canal para verificar o cache")
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("clear")
                .setDescription("Limpar cache do canal (padrão: false)")
                .setRequired(false))
        .addBooleanOption(option =>
            option.setName("ephemeral")
                .setDescription("Apenas você pode ver o resultado (padrão: false)")
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
            const clear = interaction.options.getBoolean("clear") ?? false;
            await interaction.deferReply({ flags: ephemeral ? MessageFlags.Ephemeral : undefined });

            const channel = interaction.options.getChannel("channel") || interaction.channel;
            
            if (!channel) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.info, "Por favor, forneça um canal para verificar o cache.", false));
            }

            if (!(channel instanceof TextChannel)) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.warning, "O canal deve ser um canal de texto.", false));
            }

            if (!channel.permissionsFor(interaction.client.user!)?.has(PermissionFlagsBits.ViewChannel)) {
                return await interaction.editReply(UIService.formatStatusMessage(EMOJIS.error, "Eu não tenho permissão para ver esse canal.", false));
            }

            // Get cache size and actual file size
            const cacheSize = MessageService.getCacheSize(channel.id);
            
            if (clear) {
                MessageService.clearCache(channel.id);
                await interaction.editReply(UIService.formatStatusMessage(
                    EMOJIS.delete,
                    `Cache do canal ${channel} foi limpo. (${cacheSize} mensagens removidas)`,
                    false
                ));
                return;
            }

            if (cacheSize === 0) {
                await interaction.editReply(UIService.formatStatusMessage(
                    EMOJIS.info,
                    `Não há cache para o canal ${channel}. Execute uma pesquisa ou use o contexto no canal para criar cache.`,
                    false
                ));
                return;
            }

            // Get actual file size
            const stats = FileSystemService.getFileStatsFromPath(`${channel.id}.json`, MessageService['SERVICE_NAME'], 'cache');
            const channelCacheSizeMB = stats ? (stats.size / (1024 * 1024)).toFixed(2) : '0';

            // Get total cache usage
            const cacheDir = FileSystemService.getDir(MessageService['SERVICE_NAME'], 'cache');
            const currentDiskSize = FileSystemService.getDirectorySize(cacheDir);
            const totalCacheSizeMB = (currentDiskSize / (1024 * 1024)).toFixed(2);
            const maxCacheSizeMB = (MessageService['MAX_CACHE_SIZE'] / (1024 * 1024)).toFixed(2);

            const messageHeader = `\`${EMOJIS.cache} Cache do canal \` **\`${channel.name}\`**`
            const messageBody = `Mensagens em cache: **${cacheSize.toLocaleString()}**\nTamanho do cache: **${channelCacheSizeMB} MB**\nUso total do cache: **${totalCacheSizeMB}/${maxCacheSizeMB} MB**`;
            const messageFooter = `Use \`/cache clear\` para limpar o cache deste canal.`;

            await interaction.editReply({
                content: `${messageHeader}\n\n${messageBody}\n\n${messageFooter}`,
            });

        } catch (error) {
            console.error('Error in cache command:', error);
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.error, "Ocorreu um erro ao verificar o cache. Por favor, tente novamente mais tarde.", false)
            );
        }
    },
};
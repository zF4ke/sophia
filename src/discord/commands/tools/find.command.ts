import {
    ChannelType,
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { GroundedResultUIService } from "@/discord/ui/GroundedResultUIService";
import { SecurityService } from "@/security/SecurityService";
import { UIService } from "@/discord/ui/UIService";
import { EMOJIS } from "@/discord/constants";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("find")
        .setDescription("Buscar evidência armazenada do Discord")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addStringOption((option) =>
            option
                .setName("topic")
                .setDescription("Tópico ou pergunta para buscar")
                .setRequired(true)
        )
        .addChannelOption((option) =>
            option
                .setName("channel")
                .setDescription("Opcional: limitar a busca a um canal")
                .addChannelTypes(ChannelType.GuildText, ChannelType.PublicThread, ChannelType.PrivateThread)
                .setRequired(false)
        )
        .addBooleanOption((option) =>
            option
                .setName("ephemeral")
                .setDescription("Somente você pode ver o resultado")
                .setRequired(false)
        ),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        try {
            if (!SecurityService.isAdmin(interaction.user.id)) {
                await interaction.reply({
                    content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                    flags: MessageFlags.Ephemeral,
                });
                return;
            }

            const topic = interaction.options.getString("topic", true);
            const channel = interaction.options.getChannel("channel");
            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;

            await interaction.deferReply({
                flags: ephemeral ? MessageFlags.Ephemeral : undefined,
            });

            const results = await DiscordMemoryService.searchMessagesAsync(
                topic,
                {
                    guildId: interaction.guildId,
                    channelIds: channel ? [channel.id] : undefined,
                },
                12
            );

            if (!results.length) {
                await interaction.editReply(
                    UIService.formatStatusMessage(
                        EMOJIS.warning,
                        "Nenhuma evidência armazenada foi encontrada.",
                        false
                    )
                );
                return;
            }

            await GroundedResultUIService.showSearchResults(
                interaction,
                "Resultados da memória do Discord",
                topic,
                results
            );
        } catch (error) {
            console.error("Error in find command:", error);
            await interaction.editReply(
                UIService.formatStatusMessage(
                    EMOJIS.error,
                    "Ocorreu um erro durante a busca.",
                    false
                )
            );
        }
    },
};

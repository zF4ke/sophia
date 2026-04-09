import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { SecurityService } from "@/security/SecurityService";
import { EMOJIS } from "@/discord/constants";
import { buildMemoryStatusContainer } from "@/discord/commands/shared/buildMemoryStatusContainer";

export = {
    data: new SlashCommandBuilder()
        .setName("cache")
        .setDescription("Exibe estatísticas da memória local")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addBooleanOption((option) =>
            option
                .setName("ephemeral")
                .setDescription("Somente você pode ver o resultado")
                .setRequired(false)
        ),
    async execute(interaction: ChatInputCommandInteraction) {
        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                flags: MessageFlags.Ephemeral,
            });
            return;
        }

        const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
        await interaction.deferReply({
            flags: ephemeral ? MessageFlags.Ephemeral : undefined,
        });

        const stats = DiscordMemoryService.getStats();
        const states = DiscordMemoryService.getIndexState();

        await interaction.editReply({
            components: [buildMemoryStatusContainer(stats, states)],
            flags: MessageFlags.IsComponentsV2,
        });
    },
};

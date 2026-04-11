import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { RuntimeStorageService } from "@/runtime/storage/RuntimeStorageService";
import { SecurityService } from "@/security/SecurityService";
import { EMOJIS } from "@/discord/constants";
import { buildMemoryStatusContainer } from "@/discord/commands/shared/buildMemoryStatusContainer";
import { buildRuntimeStorageContainer } from "@/discord/commands/shared/buildRuntimeStorageContainer";

export = {
    data: new SlashCommandBuilder()
        .setName("cache")
        .setDescription("Show local Discord retrieval cache stats")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addBooleanOption((option) =>
            option
                .setName("ephemeral")
                .setDescription("Only you can see the result")
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

        const stats = await DiscordMemoryService.getStatsAsync();
        const states = await DiscordMemoryService.getIndexStateAsync();
        const runtimeStatus = RuntimeStorageService.getStatus();

        await interaction.editReply({
            components: [
                buildMemoryStatusContainer(stats, states),
                buildRuntimeStorageContainer(runtimeStatus),
            ],
            flags: MessageFlags.IsComponentsV2,
        });
    },
};

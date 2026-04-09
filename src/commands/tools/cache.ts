import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DiscordMemoryService } from "@/services/memory/DiscordMemoryService";
import { SecurityService } from "@/services/SecurityService";
import { UIService } from "@/services/UIService";
import { EMOJIS } from "@/utils/constants";
import type { BotClient } from "@/types/app";

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
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
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
        const states = DiscordMemoryService.getIndexState().slice(0, 10);
        const lines = states.length
            ? states.map((state) => `• ${state.channelId}: ${state.lastMessageId || "sem cursor"}`)
            : ["• Nenhum canal indexado."];

        await interaction.editReply({
            content:
                `${UIService.formatStatusMessage(EMOJIS.memory, "Estado da memória local", false)}\n\n` +
                `Mensagens: **${stats.messages}**\n` +
                `Chunks: **${stats.chunks}**\n` +
                `Canais: **${stats.channels}**\n\n` +
                lines.join("\n"),
        });
    },
};

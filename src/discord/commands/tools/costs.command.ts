import { MessageFlags, SlashCommandBuilder, type ChatInputCommandInteraction } from "discord.js";
import { buildCostsPanel } from "../system/costsPanel";
export = {
    data: new SlashCommandBuilder().setName("costs").setDescription("Mostrar a tua utilização e custos estimados no canal").setContexts(0, 1, 2).setIntegrationTypes(0)
        .addBooleanOption(option => option.setName("ephemeral").setDescription("Mostrar o relatório só para ti")),
    async execute(interaction: ChatInputCommandInteraction) {
        await interaction.deferReply({ flags: interaction.options.getBoolean("ephemeral") ? MessageFlags.Ephemeral : undefined });
        await interaction.editReply(await buildCostsPanel(interaction.user.id, "own"));
    },
};

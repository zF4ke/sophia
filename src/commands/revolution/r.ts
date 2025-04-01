import { SlashCommandBuilder, PermissionFlagsBits, ChatInputCommandInteraction, MessageFlags } from "discord.js";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("r")
        .setDescription("🏴‍☠️")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1),
    async execute(interaction: ChatInputCommandInteraction) {        
        await interaction.reply({
            content: "🏴‍☠️"
        });
    },
};

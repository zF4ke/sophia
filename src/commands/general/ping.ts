import { SlashCommandBuilder, PermissionFlagsBits, ChatInputCommandInteraction } from "discord.js";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("ping")
        .setDescription("Pong")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        ,
    async execute(interaction: ChatInputCommandInteraction) {
        const ms = Date.now() - interaction.createdTimestamp;
        const ping = Math.round(interaction.client.ws.ping);

        await interaction.reply({
            embeds: [
                {
                    title: "🏓 Pong!",
                    description: `⏱️ Latency: \`${ms}ms\`\n🌐 API Latency: \`${ping}ms\``,
                    // make color close to the color of the discord logo
                    color: 0x7289da,
                },
            ],
            ephemeral: false,
        });
    },
};

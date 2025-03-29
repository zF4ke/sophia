import { TDiscordClient } from "@/index";
import { Interaction } from "discord.js";

module.exports = {
    name: "interactionCreate",
    async execute(interaction: Interaction, client: TDiscordClient) {
        if (interaction.isChatInputCommand()) {
            const command = client.commands.get(interaction.commandName);
            if (!command) {
                return interaction.reply({
                    ephemeral: true,
                    content: "Outdated command",
                });
            }
            command.execute(interaction, client);
        }
    },
};

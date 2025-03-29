import { TDiscordClient } from "@/index";
import { Interaction, MessageFlags } from "discord.js";

module.exports = {
    name: "interactionCreate",
    async execute(interaction: Interaction, client: TDiscordClient) {
        if (interaction.isChatInputCommand()) {
            const command = client.commands.get(interaction.commandName);
            if (!command) {
                return interaction.reply({
                    flags: MessageFlags.Ephemeral,
                    content: "Outdated command",
                });
            }
            command.execute(interaction, client);
        }
    },
};

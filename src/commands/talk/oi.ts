import { busy } from "@/utils/message";
import { SlashCommandBuilder, PermissionFlagsBits, ChatInputCommandInteraction } from "discord.js";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("oi")
        .setDescription("oi")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        ,
    async execute(interaction: ChatInputCommandInteraction) {
        try {
            throw new Error("Not implemented yet");
        } catch (error) {
            await cant(interaction);
        }
    },
};

async function cant(interaction: ChatInputCommandInteraction) {
    const randomIndex = Math.floor(Math.random() * busy.length);
    const randomMessage = busy[randomIndex];

    await interaction.reply({
        content: randomMessage,
    });
}
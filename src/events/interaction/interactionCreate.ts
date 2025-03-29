import { TDiscordClient } from "@/index";
import { Interaction, MessageFlags } from "discord.js";
import { SecurityService } from "../../services/SecurityService";

module.exports = {
    name: "interactionCreate",
    async execute(interaction: Interaction, client: TDiscordClient) {
        if (interaction.isChatInputCommand()) {
            const command = client.commands.get(interaction.commandName);
            if (!command) {
                return interaction.reply({
                    flags: MessageFlags.Ephemeral,
                    content: "Comando desatualizado",
                });
            }
            
            // Initialize security service
            await SecurityService.initialize();
            
            // Skip security checks for admin command
            if (interaction.commandName !== 'admin') {
                // Check if command is public or user is admin
                const isPublic = await SecurityService.isCommandPublic(interaction.commandName);
                if (!isPublic && !SecurityService.isAdmin(interaction.user.id)) {
                    return interaction.reply({
                        flags: MessageFlags.Ephemeral,
                        content: "❌ Este comando é restrito apenas para administradores.",
                    });
                }
                
                // Check if user has exceeded rate limit
                // Use internal moderator check instead of Discord permissions
                if (!await SecurityService.checkRateLimit(interaction.user.id, interaction.commandName)) {
                    const remainingTime = Math.ceil((SecurityService.RATE_LIMIT_WINDOW / 1000) / 60);
                    return interaction.reply({
                        flags: MessageFlags.Ephemeral,
                        content: `❌ Você atingiu o limite de uso para este comando. Por favor, aguarde ${remainingTime} minuto(s) antes de tentar novamente.`,
                    });
                }
            }
            
            // Execute command
            command.execute(interaction, client);
        }
    },
};

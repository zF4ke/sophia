import type { BotClient } from "@/shared/appTypes";
import { Interaction, MessageFlags } from "discord.js";
import { handleAccessPanelInteraction } from "@/discord/commands/system/access/panelInteractions";
import { handleDebugPanelInteraction } from "@/discord/debug/debugPanelInteractions";
import { SecurityService } from "@/security/SecurityService";

export = {
    name: "interactionCreate",
    async execute(interaction: Interaction, client: BotClient) {
        if (interaction.isAutocomplete()) {
            const command = client.commands.get(interaction.commandName);
            if (command?.autocomplete) {
                await command.autocomplete(interaction, client);
            }
            return;
        }

        if (
            interaction.isButton() ||
            interaction.isStringSelectMenu() ||
            interaction.isUserSelectMenu() ||
            interaction.isModalSubmit()
        ) {
            if (interaction.isButton()) {
                if (await handleDebugPanelInteraction(interaction)) {
                    return;
                }
            }

            if (await handleAccessPanelInteraction(interaction, client)) {
                return;
            }
        }

        if (interaction.isChatInputCommand()) {
            const command = client.commands.get(interaction.commandName);
            if (!command) {
                return interaction.reply({
                    flags: MessageFlags.Ephemeral,
                    content: "Comando desatualizado",
                });
            }
            
            await SecurityService.initialize();
            
            if (interaction.commandName !== 'access') {
                const isPublic = await SecurityService.isCommandPublic(interaction.commandName);
                if (!isPublic && !SecurityService.isAdmin(interaction.user.id)) {
                    return interaction.reply({
                        flags: MessageFlags.Ephemeral,
                        content: "❌ Este comando é restrito apenas para administradores.",
                    });
                }
                
                if (!await SecurityService.checkRateLimit(interaction.user.id, interaction.commandName)) {
                    const remainingTime = Math.ceil((SecurityService.RATE_LIMIT_WINDOW / 1000) / 60);
                    return interaction.reply({
                        flags: MessageFlags.Ephemeral,
                        content: `❌ Você atingiu o limite de uso para este comando. Por favor, aguarde ${remainingTime} minuto(s) antes de tentar novamente.`,
                    });
                }
            }
            
            await command.execute(interaction, client);
        }
    },
};

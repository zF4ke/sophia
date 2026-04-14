import type { BotClient } from "@/shared/appTypes";
import { Interaction, MessageFlags } from "discord.js";
import { handleAccessPanelInteraction } from "@/discord/commands/system/access/panelInteractions";
import { handleApprovalInteraction } from "@/discord/approval/approvalInteractions";
import { handleDebugLogsInteraction } from "@/discord/commands/system/debugLogsInteractions";
import { handleDebugPanelInteraction } from "@/discord/debug/debugPanelInteractions";
import { handleSettingsPanelInteraction } from "@/discord/commands/system/settings/settingsInteractions";
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
            if (interaction.isButton() || interaction.isModalSubmit()) {
                if (await handleApprovalInteraction(interaction)) {
                    return;
                }
            }

            if (interaction.isButton()) {
                if (await handleDebugPanelInteraction(interaction)) {
                    return;
                }
            }

            if (
                interaction.isButton() ||
                interaction.isStringSelectMenu()
            ) {
                if (await handleDebugLogsInteraction(interaction)) {
                    return;
                }
            }

            if (
                interaction.isButton() ||
                interaction.isStringSelectMenu()
            ) {
                if (await handleSettingsPanelInteraction(interaction)) {
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
                    content: "Unknown command.",
                });
            }
            
            await SecurityService.initialize();
            
            if (interaction.commandName !== 'access') {
                const isPublic = await SecurityService.isCommandPublic(interaction.commandName);
                if (!isPublic && !SecurityService.isAdmin(interaction.user.id)) {
                    return interaction.reply({
                        flags: MessageFlags.Ephemeral,
                        content: "❌ This command is restricted to administrators.",
                    });
                }
                
                if (!await SecurityService.checkRateLimit(interaction.user.id, interaction.commandName)) {
                    const remainingTime = Math.ceil((SecurityService.RATE_LIMIT_WINDOW / 1000) / 60);
                    return interaction.reply({
                        flags: MessageFlags.Ephemeral,
                        content: `❌ Rate limit reached for this command. Please wait ${remainingTime} minute(s) before trying again.`,
                    });
                }
            }
            
            await command.execute(interaction, client);
        }
    },
};

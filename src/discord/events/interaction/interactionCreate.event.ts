import type { BotClient } from "@/shared/appTypes";
import { Interaction, MessageFlags } from "discord.js";
import { handleAccessPanelInteraction } from "@/discord/commands/system/access/panelInteractions";
import { handleApprovalInteraction } from "@/discord/approval/approvalInteractions";
import { handleArtifactInteraction } from "@/discord/artifacts/ArtifactInteractions";
import { handleDebugLogsInteraction } from "@/discord/commands/system/debugLogsInteractions";
import { handleDebugPanelInteraction } from "@/discord/debug/debugPanelInteractions";
import { handleSettingsPanelInteraction } from "@/discord/commands/system/settings/settingsInteractions";
import { handleUiTestSettingsInteraction } from "@/discord/commands/system/uitest/settings";
import { SecurityService } from "@/security/SecurityService";
import { isGuildAllowed } from "@/security/guildAllowlist";

export = {
    name: "interactionCreate",
    async execute(interaction: Interaction, client: BotClient) {
        if (!isGuildAllowed(interaction.guildId)) return;

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
            // Artifact card controls are prefix-matched and DB-backed; check
            // them first so they resolve even after a restart.
            if (interaction.isButton() || interaction.isStringSelectMenu()) {
                if (await handleArtifactInteraction(interaction)) {
                    return;
                }
            }

            if (interaction.isButton() || interaction.isModalSubmit() || interaction.isStringSelectMenu()) {
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

            if (
                interaction.isButton() ||
                interaction.isStringSelectMenu()
            ) {
                if (await handleUiTestSettingsInteraction(interaction)) {
                    return;
                }
            }

            if (await handleAccessPanelInteraction(interaction, client)) {
                return;
            }

            // Nothing claimed this component interaction. Acknowledge it
            // anyway so the client never shows "didn't respond in time" for
            // buttons that belong to stale or foreign panels.
            if (interaction.isButton() || interaction.isStringSelectMenu()) {
                await interaction.deferUpdate().catch(() => undefined);
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

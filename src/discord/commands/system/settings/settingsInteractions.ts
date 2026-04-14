import { ButtonInteraction, MessageFlags, StringSelectMenuInteraction } from "discord.js";
import { SettingsService } from "@/app/SettingsService";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import {
    handleSettingsInteraction,
    buildSettingsPanel,
    buildRuntimeValueSelect,
    type RuntimeSettingKey,
} from "@/discord/commands/system/settings/settings.command";
import { SecurityService } from "@/security/SecurityService";

export async function handleSettingsPanelInteraction(
    interaction: ButtonInteraction | StringSelectMenuInteraction
): Promise<boolean> {
    const customId = interaction.customId;
    if (!customId.startsWith("settings:")) return false;

    await SecurityService.initialize();
    if (!SecurityService.isAdmin(interaction.user.id)) {
        await interaction.reply({
            content: "❌ You don't have permission to change settings.",
            flags: MessageFlags.Ephemeral as const,
        });
        return true;
    }

    const parsed = handleSettingsInteraction(customId);

    try {
        switch (parsed.type) {
            case "profile": {
                if (!interaction.isStringSelectMenu()) break;
                const profile = interaction.values[0];
                const settings = SettingsService.update({ modelProfile: profile });
                await interaction.update(buildSettingsPanel(settings));
                break;
            }
            case "runtime_pick": {
                if (!interaction.isStringSelectMenu()) break;
                const key = interaction.values[0] as RuntimeSettingKey;
                const settings = SettingsService.load();
                const currentValue = settings.runtime[key];
                await interaction.update(buildRuntimeValueSelect(key, currentValue));
                break;
            }
            case "runtime_set": {
                if (!interaction.isStringSelectMenu() || !parsed.key) break;
                const key = parsed.key as RuntimeSettingKey;
                const numValue = Number(interaction.values[0]);
                const current = SettingsService.load();
                const updated = SettingsService.update({
                    runtime: { ...current.runtime, [key]: numValue },
                });
                await interaction.update(buildSettingsPanel(updated));
                break;
            }
            case "debug_toggle": {
                const settings = SettingsService.load();
                const newDebug = !settings.debug;
                DebugModeService.setEnabled(newDebug);
                const updated = SettingsService.load();
                await interaction.update(buildSettingsPanel(updated));
                break;
            }
            case "reset": {
                const settings = SettingsService.reset();
                DebugModeService.resetForTests();
                await interaction.update(buildSettingsPanel(settings));
                break;
            }
            default:
                return false;
        }
    } catch (error) {
        console.error("Settings interaction error:", error);
        if (!interaction.replied && !interaction.deferred) {
            await interaction.reply({
                content: "❌ Failed to update settings.",
                flags: MessageFlags.Ephemeral as const,
            });
        }
    }

    return true;
}

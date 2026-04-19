import { ButtonInteraction, MessageFlags, StringSelectMenuInteraction } from "discord.js";
import { SettingsService, type BotSettings } from "@/app/SettingsService";
import {
    handleSettingsInteraction,
    buildSettingsPanel,
    buildRuntimeValueSelect,
    getRuntimeSettingValue,
    buildRuntimeSettingPatch,
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
            content: "❌ Apenas administradores podem alterar definições.",
            flags: MessageFlags.Ephemeral as const,
        });
        return true;
    }

    const parsed = handleSettingsInteraction(customId);

    try {
        switch (parsed.type) {
            case "tab_model": {
                const settings = SettingsService.load();
                await interaction.update(buildSettingsPanel(settings, "model"));
                break;
            }
            case "tab_runtime": {
                const settings = SettingsService.load();
                await interaction.update(buildSettingsPanel(settings, "runtime"));
                break;
            }
            case "tab_compaction": {
                const settings = SettingsService.load();
                await interaction.update(buildSettingsPanel(settings, "compaction"));
                break;
            }
            case "tab_long_task": {
                const settings = SettingsService.load();
                await interaction.update(buildSettingsPanel(settings, "longTask"));
                break;
            }
            case "tab_personality": {
                const settings = SettingsService.load();
                await interaction.update(buildSettingsPanel(settings, "personality"));
                break;
            }
            case "personality_select": {
                if (!interaction.isStringSelectMenu()) break;
                const raw = interaction.values[0];
                const allowed: BotSettings["personality"][] = ["default", "mixed", "classic"];
                const next = (allowed as string[]).includes(raw)
                    ? (raw as BotSettings["personality"])
                    : "default";
                const updated = SettingsService.update({ personality: next });
                await interaction.update(buildSettingsPanel(updated, "personality"));
                break;
            }
            case "model_select": {
                if (!interaction.isStringSelectMenu()) break;
                const profile = interaction.values[0];
                const settings = SettingsService.update({ modelProfile: profile });
                await interaction.update(buildSettingsPanel(settings, "model"));
                break;
            }
            case "compaction_model_select": {
                if (!interaction.isStringSelectMenu()) break;
                const selected = interaction.values[0];
                if (selected === "_none") break;
                const current = SettingsService.load();
                const updated = SettingsService.update({
                    compaction: { ...current.compaction, summarizerModel: selected },
                });
                await interaction.update(buildSettingsPanel(updated, "compaction"));
                break;
            }
            case "runtime_pick": {
                if (!interaction.isStringSelectMenu()) break;
                const key = interaction.values[0] as RuntimeSettingKey;
                const settings = SettingsService.load();
                const currentValue = getRuntimeSettingValue(settings, key);
                await interaction.update(buildRuntimeValueSelect(key, currentValue));
                break;
            }
            case "runtime_set": {
                if (!interaction.isStringSelectMenu() || !parsed.key) break;
                const key = parsed.key as RuntimeSettingKey;
                const numValue = Number(interaction.values[0]);
                const current = SettingsService.load();
                const updated = SettingsService.update(
                    buildRuntimeSettingPatch(current, key, numValue)
                );
                const backTab = key.startsWith("longTask") ? "longTask" : "runtime";
                await interaction.update(buildSettingsPanel(updated, backTab));
                break;
            }
            case "auto_approve_writes_toggle": {
                const current = SettingsService.load();
                const updated = SettingsService.update({
                    runtime: {
                        ...current.runtime,
                        autoApproveWrites: !current.runtime.autoApproveWrites,
                    },
                });
                await interaction.update(buildSettingsPanel(updated, "runtime"));
                break;
            }
            case "reset": {
                const settings = SettingsService.reset();
                await interaction.update(buildSettingsPanel(settings, "model"));
                break;
            }
            default:
                return false;
        }
    } catch (error) {
        console.error("Settings interaction error:", error);
        if (!interaction.replied && !interaction.deferred) {
            await interaction.reply({
                content: "❌ Falha ao atualizar definições.",
                flags: MessageFlags.Ephemeral as const,
            });
        }
    }

    return true;
}

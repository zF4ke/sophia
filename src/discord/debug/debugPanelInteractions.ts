import { ButtonInteraction, MessageFlags } from "discord.js";
import {
    DEBUG_DISABLE_ID,
    DEBUG_ENABLE_ID,
    renderDebugControlPanel,
} from "@/discord/debug/renderDebugControlPanel";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import { SecurityService } from "@/security/SecurityService";

export async function handleDebugPanelInteraction(
    interaction: ButtonInteraction
): Promise<boolean> {
    if (
        interaction.customId !== DEBUG_ENABLE_ID &&
        interaction.customId !== DEBUG_DISABLE_ID
    ) {
        return false;
    }

    await SecurityService.initialize();

    if (!SecurityService.isAdmin(interaction.user.id)) {
        await interaction.reply({
            content: "❌ Você não tem permissão para usar este painel.",
        });
        return true;
    }

    const enabled = interaction.customId === DEBUG_ENABLE_ID;
    DebugModeService.setEnabled(enabled);

    await interaction.update({
        ...renderDebugControlPanel(
            enabled,
            enabled ? "Debug global ativado." : "Debug global desativado."
        ),
        flags: MessageFlags.IsComponentsV2,
    });

    return true;
}

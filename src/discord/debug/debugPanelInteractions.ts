import { ButtonInteraction, MessageFlags } from "discord.js";
import {
    DEBUG_DISABLE_ID,
    DEBUG_ENABLE_ID,
    renderDebugControlPanel,
} from "@/discord/debug/renderDebugControlPanel";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import { SecurityService } from "@/security/SecurityService";

function isUnknownInteractionError(error: unknown): boolean {
    return Boolean(
        error &&
            typeof error === "object" &&
            "code" in error &&
            (error as { code?: unknown }).code === 10062
    );
}

export async function handleDebugPanelInteraction(
    interaction: ButtonInteraction
): Promise<boolean> {
    const isModeToggle =
        interaction.customId === DEBUG_ENABLE_ID ||
        interaction.customId === DEBUG_DISABLE_ID;

    if (!isModeToggle) {
        return false;
    }

    await SecurityService.initialize();

    if (!SecurityService.isAdmin(interaction.user.id)) {
        try {
            await interaction.reply({
                content: "❌ You don't have permission to use this panel.",
            });
        } catch (error) {
            if (!isUnknownInteractionError(error)) {
                throw error;
            }
        }
        return true;
    }

    const enabled = interaction.customId === DEBUG_ENABLE_ID;
    DebugModeService.setEnabled(enabled);

    await interaction.update({
        ...renderDebugControlPanel(
            enabled,
            enabled ? "Debug mode enabled." : "Debug mode disabled."
        ),
        flags: MessageFlags.IsComponentsV2,
    });

    return true;
}

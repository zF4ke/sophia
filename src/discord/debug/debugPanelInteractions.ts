import { ButtonInteraction, MessageFlags } from "discord.js";
import { DebugSession } from "@/discord/debug/DebugSession";
import {
    DEBUG_DISABLE_ID,
    DEBUG_ENABLE_ID,
    renderDebugControlPanel,
} from "@/discord/debug/renderDebugControlPanel";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import {
    isDebugTraceCollapseAll,
    isDebugTraceExpandAll,
    parseDebugTraceSectionToggle,
} from "@/discord/debug/renderDebugTrace";
import { SecurityService } from "@/security/SecurityService";

function isUnknownInteractionError(error: unknown): boolean {
    return Boolean(
        error &&
            typeof error === "object" &&
            "code" in error &&
            (error as { code?: unknown }).code === 10062
    );
}

async function safeDeferUpdate(interaction: ButtonInteraction): Promise<boolean> {
    try {
        if (!interaction.deferred && !interaction.replied) {
            await interaction.deferUpdate();
        }
        return true;
    } catch (error) {
        if (isUnknownInteractionError(error)) {
            return false;
        }
        throw error;
    }
}

export async function handleDebugPanelInteraction(
    interaction: ButtonInteraction
): Promise<boolean> {
    const toggledSection = parseDebugTraceSectionToggle(interaction.customId);
    const traceExpandAll = isDebugTraceExpandAll(interaction.customId);
    const traceCollapseAll = isDebugTraceCollapseAll(interaction.customId);
    const isModeToggle =
        interaction.customId === DEBUG_ENABLE_ID ||
        interaction.customId === DEBUG_DISABLE_ID;

    if (!isModeToggle && !toggledSection && !traceExpandAll && !traceCollapseAll) {
        return false;
    }

    await SecurityService.initialize();

    if (!SecurityService.isAdmin(interaction.user.id)) {
        try {
            await interaction.reply({
                content: "❌ Você não tem permissão para usar este painel.",
            });
        } catch (error) {
            if (!isUnknownInteractionError(error)) {
                throw error;
            }
        }
        return true;
    }

    if (toggledSection || traceExpandAll || traceCollapseAll) {
        const session = DebugSession.getByMessageId(interaction.message.id);
        if (!session) {
            try {
                await interaction.reply({
                    content: "⚠️ Esta sessão de debug expirou e não pode mais ser recolhida.",
                    flags: MessageFlags.Ephemeral,
                });
            } catch (error) {
                if (!isUnknownInteractionError(error)) {
                    throw error;
                }
            }
            return true;
        }

        const acknowledged = await safeDeferUpdate(interaction);
        if (!acknowledged) {
            return true;
        }

        if (toggledSection) {
            await session.toggleSection(toggledSection);
        } else if (traceExpandAll) {
            await session.setAllSectionsCollapsed(false);
        } else {
            await session.setAllSectionsCollapsed(true);
        }

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

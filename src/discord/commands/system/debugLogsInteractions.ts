import { ButtonInteraction, MessageFlags, StringSelectMenuInteraction } from "discord.js";
import {
    buildDebugLogsPanel,
    DEBUG_LOGS_CONTENT_PAGE_NEXT_ID,
    DEBUG_LOGS_CONTENT_PAGE_PREV_ID,
    DEBUG_LOGS_ENTRY_PAGE_NEXT_ID,
    DEBUG_LOGS_ENTRY_PAGE_PREV_ID,
    DEBUG_LOGS_ENTRY_SELECT_ID,
    DEBUG_LOGS_FILE_SELECT_ID,
    DEBUG_LOGS_VIEW_SELECT_ID,
    getDebugLogsPanelState,
    rememberDebugLogsPanelState,
    type DebugLogsPanelState,
    type DebugLogsView,
} from "@/discord/commands/system/debugLogsPanel";
import { SecurityService } from "@/security/SecurityService";

function isPanelInteraction(customId: string): boolean {
    return customId.startsWith("debug:logs:");
}

function withUpdatedState(state: DebugLogsPanelState): DebugLogsPanelState {
    return {
        ...state,
        updatedAt: Date.now(),
    };
}

export async function handleDebugLogsInteraction(
    interaction: ButtonInteraction | StringSelectMenuInteraction
): Promise<boolean> {
    if (!isPanelInteraction(interaction.customId)) {
        return false;
    }

    await SecurityService.initialize();
    if (!SecurityService.isAdmin(interaction.user.id)) {
        await interaction.reply({
            content: "❌ Apenas administradores podem usar este painel.",
            flags: MessageFlags.Ephemeral as const,
        });
        return true;
    }

    const state = getDebugLogsPanelState(interaction.message.id);
    if (!state) {
        await interaction.reply({
            content: "❌ Este painel de logs expirou. Executa /debug logs novamente.",
            flags: MessageFlags.Ephemeral as const,
        });
        return true;
    }

    let nextState = { ...state };

    if (interaction.isStringSelectMenu()) {
        if (interaction.customId === DEBUG_LOGS_FILE_SELECT_ID) {
            nextState = withUpdatedState({
                ...nextState,
                selectedFile: interaction.values[0] || nextState.selectedFile,
                selectedEntryIndex: 0,
                entryListPage: 0,
                activeView: "overview",
                contentPage: 0,
            });
        } else if (interaction.customId === DEBUG_LOGS_ENTRY_SELECT_ID) {
            const selectedEntryIndex = Number(interaction.values[0] || 0);
            nextState = withUpdatedState({
                ...nextState,
                selectedEntryIndex,
                entryListPage: Math.floor(Math.max(0, selectedEntryIndex) / 25),
                contentPage: 0,
            });
        } else if (interaction.customId === DEBUG_LOGS_VIEW_SELECT_ID) {
            nextState = withUpdatedState({
                ...nextState,
                activeView: (interaction.values[0] as DebugLogsView) || nextState.activeView,
                contentPage: 0,
            });
        }
    }

    if (interaction.isButton()) {
        if (interaction.customId === DEBUG_LOGS_ENTRY_PAGE_PREV_ID) {
            const entryListPage = Math.max(0, nextState.entryListPage - 1);
            nextState = withUpdatedState({
                ...nextState,
                entryListPage,
                selectedEntryIndex: entryListPage * 25,
                contentPage: 0,
            });
        } else if (interaction.customId === DEBUG_LOGS_ENTRY_PAGE_NEXT_ID) {
            const entryListPage = nextState.entryListPage + 1;
            nextState = withUpdatedState({
                ...nextState,
                entryListPage,
                selectedEntryIndex: entryListPage * 25,
                contentPage: 0,
            });
        } else if (interaction.customId === DEBUG_LOGS_CONTENT_PAGE_PREV_ID) {
            nextState = withUpdatedState({
                ...nextState,
                contentPage: Math.max(0, nextState.contentPage - 1),
            });
        } else if (interaction.customId === DEBUG_LOGS_CONTENT_PAGE_NEXT_ID) {
            nextState = withUpdatedState({
                ...nextState,
                contentPage: nextState.contentPage + 1,
            });
        }
    }

    rememberDebugLogsPanelState(interaction.message.id, nextState);
    await interaction.update(buildDebugLogsPanel(nextState));
    return true;
}

import { loadAccessPanelData } from "@/discord/commands/system/access/panelData";
import { buildAccessPanel } from "@/discord/commands/system/access/panelRenderer";
import type {
    AccessInteraction,
    AccessHandlerContext,
} from "@/discord/commands/system/access/handlers/types";
import type {
    AccessPanelState,
    AccessPanelView,
} from "@/discord/commands/system/access/panelTypes";

export async function refreshPanel(
    interaction: AccessInteraction,
    context: AccessHandlerContext,
    state: AccessPanelState
): Promise<void> {
    const payload = await buildAccessPanel(
        context.client,
        await loadAccessPanelData(context.client),
        state
    );

    if (interaction.isModalSubmit()) {
        await interaction.message?.edit(payload);
        await interaction.reply({
            content: state.notice ?? "Painel atualizado.",
            ephemeral: interaction.message?.flags.has("Ephemeral") ?? false,
        });
        return;
    }

    await interaction.update(payload);
}

export function getFallbackView(customId: string): AccessPanelView {
    const parts = customId.split(":");
    if (parts[1] === "admins") {
        return "admins";
    }
    if (parts[1] === "moderators") {
        return "moderators";
    }
    if (parts[1] === "commands") {
        return "commands";
    }
    return "overview";
}

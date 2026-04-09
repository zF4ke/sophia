import { isAccessInteraction } from "./panelIds";
import type { BotClient } from "@/shared/appTypes";
import { rejectNonAdmin } from "@/discord/commands/system/access/handlers/adminGuard";
import { handleAccessButton } from "@/discord/commands/system/access/handlers/buttonHandlers";
import { handleAccessModal } from "@/discord/commands/system/access/handlers/modalHandlers";
import { getFallbackView, refreshPanel } from "@/discord/commands/system/access/handlers/panelRefresh";
import {
    handleAccessStringSelect,
    handleAccessUserSelect,
} from "@/discord/commands/system/access/handlers/selectHandlers";
import type { AccessHandlerContext, AccessInteraction } from "@/discord/commands/system/access/handlers/types";

export async function handleAccessPanelInteraction(
    interaction: AccessInteraction,
    client: BotClient
): Promise<boolean> {
    if (!isAccessInteraction(interaction.customId)) {
        return false;
    }

    if (await rejectNonAdmin(interaction)) {
        return true;
    }

    const context: AccessHandlerContext = { client };

    if (interaction.isButton()) {
        if (await handleAccessButton(interaction, context)) {
            return true;
        }
    }

    if (interaction.isUserSelectMenu()) {
        if (await handleAccessUserSelect(interaction, context)) {
            return true;
        }
    }

    if (interaction.isStringSelectMenu()) {
        if (await handleAccessStringSelect(interaction, context)) {
            return true;
        }
    }

    if (interaction.isModalSubmit()) {
        if (await handleAccessModal(interaction, context)) {
            return true;
        }
    }

    await refreshPanel(interaction, context, {
        view: getFallbackView(interaction.customId),
    });
    return true;
}

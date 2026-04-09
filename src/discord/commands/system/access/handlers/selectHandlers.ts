import {
    StringSelectMenuInteraction,
    UserSelectMenuInteraction,
} from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import {
    ACCESS_ADMIN_ADD_ID,
    ACCESS_ADMIN_REMOVE_ID,
    ACCESS_COMMAND_SELECT_ID,
    ACCESS_MODERATOR_ADD_ID,
    ACCESS_MODERATOR_REMOVE_ID,
} from "@/discord/commands/system/access/panelIds";
import { refreshPanel } from "@/discord/commands/system/access/handlers/panelRefresh";
import type { AccessHandlerContext } from "@/discord/commands/system/access/handlers/types";

export async function handleAccessUserSelect(
    interaction: UserSelectMenuInteraction,
    context: AccessHandlerContext
): Promise<boolean> {
    const targetUserId = interaction.values[0];

    if (interaction.customId === ACCESS_ADMIN_ADD_ID) {
        const added = await SecurityService.addAdmin(targetUserId, interaction.user.id);
        await refreshPanel(interaction, context, {
            view: "admins",
            notice: added ? "Administrador adicionado." : "Esse usuário já é administrador.",
        });
        return true;
    }

    if (interaction.customId === ACCESS_MODERATOR_ADD_ID) {
        const added = await SecurityService.addModerator(targetUserId, interaction.user.id);
        await refreshPanel(interaction, context, {
            view: "moderators",
            notice: added
                ? "Moderador adicionado."
                : "Esse usuário já é moderador ou administrador.",
        });
        return true;
    }

    return false;
}

export async function handleAccessStringSelect(
    interaction: StringSelectMenuInteraction,
    context: AccessHandlerContext
): Promise<boolean> {
    if (interaction.customId === ACCESS_ADMIN_REMOVE_ID) {
        const removed = await SecurityService.removeAdmin(interaction.values[0]);
        await refreshPanel(interaction, context, {
            view: "admins",
            notice: removed
                ? "Administrador removido."
                : "Esse administrador não pode ser removido.",
        });
        return true;
    }

    if (interaction.customId === ACCESS_MODERATOR_REMOVE_ID) {
        const removed = await SecurityService.removeModerator(interaction.values[0]);
        await refreshPanel(interaction, context, {
            view: "moderators",
            notice: removed ? "Moderador removido." : "Esse usuário não é moderador.",
        });
        return true;
    }

    if (interaction.customId === ACCESS_COMMAND_SELECT_ID) {
        await refreshPanel(interaction, context, {
            view: "commands",
            selectedCommand: interaction.values[0],
        });
        return true;
    }

    return false;
}

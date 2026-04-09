import { MessageFlags } from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import type { AccessInteraction } from "@/discord/commands/system/access/handlers/types";

export async function rejectNonAdmin(interaction: AccessInteraction): Promise<boolean> {
    await SecurityService.initialize();
    if (SecurityService.isAdmin(interaction.user.id)) {
        return false;
    }

    await interaction.reply({
        content: "❌ Você não tem permissão para usar este painel.",
        flags: MessageFlags.Ephemeral,
    });
    return true;
}

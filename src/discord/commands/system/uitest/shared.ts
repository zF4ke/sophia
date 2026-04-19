import { MessageFlags, type ChatInputCommandInteraction } from "discord.js";
import { SecurityService } from "@/security/SecurityService";

export async function guardUiTestAdmin(interaction: ChatInputCommandInteraction): Promise<boolean> {
    await SecurityService.initialize();

    if (SecurityService.isAdmin(interaction.user.id)) {
        return true;
    }

    await interaction.reply({
        content: "❌ Apenas administradores podem usar este comando.",
        flags: MessageFlags.Ephemeral,
    });
    return false;
}

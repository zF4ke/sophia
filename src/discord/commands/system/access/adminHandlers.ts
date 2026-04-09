import { ChatInputCommandInteraction } from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import {
    fetchUserUsername,
    replyWithAccessMessage,
    requireTargetUser,
} from "./shared";

export async function handleAdminSubcommand(
    interaction: ChatInputCommandInteraction,
    subcommand: string,
    ephemeral: boolean
): Promise<boolean> {
    if (subcommand === "list") {
        const admins = await SecurityService.getAllAdmins();
        if (!admins.length) {
            await replyWithAccessMessage(interaction, "Nenhum administrador encontrado", ephemeral);
            return true;
        }

        const adminList = await Promise.all(
            admins.map(async (admin) => {
                const username = await fetchUserUsername(interaction, admin.userId);
                const addedAt = new Date(admin.addedAt).toLocaleString();
                const addedBy = admin.addedBy === "system" ? "Sistema" : admin.addedBy;
                return `• **${username}** (Adicionado por: ${addedBy} em ${addedAt})`;
            })
        );

        await replyWithAccessMessage(
            interaction,
            `### Usuários Administradores\n${adminList.join("\n")}`,
            ephemeral,
            { allowedMentions: { parse: [] } }
        );
        return true;
    }

    const user = await requireTargetUser(interaction, ephemeral);
    if (!user) {
        return true;
    }

    if (subcommand === "add") {
        const added = await SecurityService.addAdmin(user.id, interaction.user.id);
        await replyWithAccessMessage(
            interaction,
            added
                ? `✅ **${user.username}** foi adicionado como administrador.`
                : `❌ **${user.username}** já é um administrador.`,
            ephemeral,
            { allowedMentions: { parse: [] } }
        );
        return true;
    }

    if (subcommand === "remove") {
        const removed = await SecurityService.removeAdmin(user.id);
        await replyWithAccessMessage(
            interaction,
            removed
                ? `✅ **${user.username}** foi removido dos administradores.`
                : `❌ **${user.username}** não pôde ser removido. Pode ser um administrador fixo ou não é um administrador.`,
            ephemeral,
            { allowedMentions: { parse: [] } }
        );
        return true;
    }

    return false;
}

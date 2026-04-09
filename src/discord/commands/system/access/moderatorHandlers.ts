import { ChatInputCommandInteraction } from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import {
    fetchUserUsername,
    replyWithAccessMessage,
    requireTargetUser,
} from "./shared";

export async function handleModeratorSubcommand(
    interaction: ChatInputCommandInteraction,
    subcommand: string,
    ephemeral: boolean
): Promise<boolean> {
    if (subcommand === "list") {
        const moderators = await SecurityService.getAllModerators();
        if (!moderators.length) {
            await replyWithAccessMessage(interaction, "Nenhum moderador encontrado", ephemeral);
            return true;
        }

        const moderatorList = await Promise.all(
            moderators.map(async (moderator) => {
                const username = await fetchUserUsername(interaction, moderator.userId);
                const addedAt = new Date(moderator.addedAt).toLocaleString();
                const addedBy =
                    moderator.addedBy === "system" ? "Sistema" : `<@${moderator.addedBy}>`;
                return `• **${username}** (Adicionado por: ${addedBy} em ${addedAt})`;
            })
        );

        await replyWithAccessMessage(
            interaction,
            `### Usuários Moderadores\n${moderatorList.join("\n")}`,
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
        const added = await SecurityService.addModerator(user.id, interaction.user.id);
        await replyWithAccessMessage(
            interaction,
            added
                ? `✅ **${user.username}** foi adicionado como moderador.`
                : `❌ **${user.username}** já é um moderador ou administrador.`,
            ephemeral,
            { allowedMentions: { parse: [] } }
        );
        return true;
    }

    if (subcommand === "remove") {
        const removed = await SecurityService.removeModerator(user.id);
        await replyWithAccessMessage(
            interaction,
            removed
                ? `✅ **${user.username}** foi removido dos moderadores.`
                : `❌ **${user.username}** não é um moderador.`,
            ephemeral,
            { allowedMentions: { parse: [] } }
        );
        return true;
    }

    return false;
}

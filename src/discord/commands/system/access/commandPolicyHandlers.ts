import { AutocompleteInteraction, ChatInputCommandInteraction } from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import { DEFAULT_ADMIN_LIMIT, DEFAULT_LIMIT, DEFAULT_MODERATOR_LIMIT } from "./constants";
import { replyWithAccessMessage } from "./shared";
import type { BotClient } from "@/shared/appTypes";

export async function handleCommandAutocomplete(
    interaction: AutocompleteInteraction,
    client: BotClient
): Promise<void> {
    const focused = interaction.options.getFocused().toLowerCase();
    const commandNames = [...client.commands.keys()]
        .filter((name) => name.startsWith(focused))
        .slice(0, 25)
        .map((name) => ({ name, value: name }));

    await interaction.respond(commandNames);
}

export async function handleCommandPolicySubcommand(
    interaction: ChatInputCommandInteraction,
    subcommand: string,
    ephemeral: boolean
): Promise<boolean> {
    const commandName = interaction.options.getString("command");

    if (subcommand === "list") {
        const commands = await SecurityService.getCommandConfigs();
        if (!commands.size) {
            await replyWithAccessMessage(interaction, "Nenhum comando encontrado", ephemeral);
            return true;
        }

        const commandList = Array.from(commands, ([name, config]) => {
            const visibility = config.isPublic ? "Público" : "Privado";
            return `• **${name}**: ${visibility} | Limites: Padrão: ${config.rateLimits.default}, Moderador: ${config.rateLimits.moderator}, Admin: ${config.rateLimits.admin}`;
        });

        await replyWithAccessMessage(
            interaction,
            `### Comandos Disponíveis\n${commandList.join("\n")}`,
            ephemeral
        );
        return true;
    }

    if (!commandName) {
        await replyWithAccessMessage(interaction, "❌ Comando não encontrado", ephemeral);
        return true;
    }

    if (subcommand === "visibility") {
        const isPublic = interaction.options.getBoolean("public", true);
        await SecurityService.setCommandVisibility(commandName, isPublic);
        await replyWithAccessMessage(
            interaction,
            `✅ Comando \`${commandName}\` agora é ${isPublic ? "público" : "privado"}.`,
            ephemeral
        );
        return true;
    }

    if (subcommand === "limit") {
        const defaultLimit = interaction.options.getInteger("default") ?? DEFAULT_LIMIT;
        const adminLimit = interaction.options.getInteger("admin") ?? DEFAULT_ADMIN_LIMIT;
        const moderatorLimit =
            interaction.options.getInteger("moderator") ?? DEFAULT_MODERATOR_LIMIT;

        await SecurityService.setCommandRateLimit(
            commandName,
            defaultLimit,
            adminLimit,
            moderatorLimit
        );
        await replyWithAccessMessage(
            interaction,
            `✅ Limites para o comando \`${commandName}\` foram definidos:\n• Padrão: ${defaultLimit}\n• Moderador: ${moderatorLimit}\n• Admin: ${adminLimit}`,
            ephemeral
        );
        return true;
    }

    return false;
}

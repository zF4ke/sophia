import {
    AutocompleteInteraction,
    ChatInputCommandInteraction,
    MessageFlags,
    PermissionFlagsBits,
    SlashCommandBuilder,
} from "discord.js";
import { handleAdminSubcommand } from "./adminHandlers";
import {
    handleCommandAutocomplete,
    handleCommandPolicySubcommand,
} from "./commandPolicyHandlers";
import { DEFAULT_EPHEMERAL } from "./constants";
import { handleModeratorSubcommand } from "./moderatorHandlers";
import { replyWithAccessMessage } from "./shared";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("access")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(1)
        .setDescription("Comandos de administração para gerenciar a segurança do bot")
        .setDefaultMemberPermissions(PermissionFlagsBits.Administrator)
        .setDMPermission(false)
        .addSubcommandGroup((group) =>
            group
                .setName("admins")
                .setDescription("Gerenciar usuários administradores")
                .addSubcommand((subcommand) =>
                    subcommand.setName("list").setDescription("Listar todos os administradores")
                )
                .addSubcommand((subcommand) =>
                    subcommand
                        .setName("add")
                        .setDescription("Adicionar um novo administrador")
                        .addUserOption((option) =>
                            option
                                .setName("user")
                                .setDescription("O usuário para adicionar como administrador")
                                .setRequired(true)
                        )
                )
                .addSubcommand((subcommand) =>
                    subcommand
                        .setName("remove")
                        .setDescription("Remover um administrador")
                        .addUserOption((option) =>
                            option
                                .setName("user")
                                .setDescription("O administrador para remover")
                                .setRequired(true)
                        )
                )
        )
        .addSubcommandGroup((group) =>
            group
                .setName("moderators")
                .setDescription("Gerenciar usuários moderadores")
                .addSubcommand((subcommand) =>
                    subcommand.setName("list").setDescription("Listar todos os moderadores")
                )
                .addSubcommand((subcommand) =>
                    subcommand
                        .setName("add")
                        .setDescription("Adicionar um novo moderador")
                        .addUserOption((option) =>
                            option
                                .setName("user")
                                .setDescription("O usuário para adicionar como moderador")
                                .setRequired(true)
                        )
                )
                .addSubcommand((subcommand) =>
                    subcommand
                        .setName("remove")
                        .setDescription("Remover um moderador")
                        .addUserOption((option) =>
                            option
                                .setName("user")
                                .setDescription("O moderador para remover")
                                .setRequired(true)
                        )
                )
        )
        .addSubcommandGroup((group) =>
            group
                .setName("command")
                .setDescription("Gerenciar configurações de comandos")
                .addSubcommand((subcommand) =>
                    subcommand
                        .setName("visibility")
                        .setDescription("Definir visibilidade do comando")
                        .addStringOption((option) =>
                            option
                                .setName("command")
                                .setDescription("O comando a ser configurado")
                                .setRequired(true)
                                .setAutocomplete(true)
                        )
                        .addBooleanOption((option) =>
                            option
                                .setName("public")
                                .setDescription("Se o comando é público")
                                .setRequired(true)
                        )
                )
                .addSubcommand((subcommand) =>
                    subcommand
                        .setName("limit")
                        .setDescription("Definir limites de uso do comando")
                        .addStringOption((option) =>
                            option
                                .setName("command")
                                .setDescription("O comando a ser configurado")
                                .setRequired(true)
                                .setAutocomplete(true)
                        )
                        .addIntegerOption((option) =>
                            option
                                .setName("default")
                                .setDescription("Limite padrão para usuários comuns")
                                .setRequired(true)
                                .setMinValue(1)
                                .setMaxValue(100)
                        )
                        .addIntegerOption((option) =>
                            option
                                .setName("admin")
                                .setDescription("Limite para administradores")
                                .setRequired(false)
                                .setMinValue(1)
                                .setMaxValue(200)
                        )
                        .addIntegerOption((option) =>
                            option
                                .setName("moderator")
                                .setDescription("Limite para moderadores")
                                .setRequired(false)
                                .setMinValue(1)
                                .setMaxValue(150)
                        )
                )
                .addSubcommand((subcommand) =>
                    subcommand.setName("list").setDescription("Listar todos os comandos")
                )
        ),
    async autocomplete(interaction: AutocompleteInteraction, client: BotClient) {
        await handleCommandAutocomplete(interaction, client);
    },
    async execute(interaction: ChatInputCommandInteraction) {
        await SecurityService.initialize();

        const ephemeral = interaction.options.getBoolean("ephemeral") ?? DEFAULT_EPHEMERAL;
        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ Você não tem permissão para usar este comando.",
                flags: ephemeral ? MessageFlags.Ephemeral : undefined,
            });
            return;
        }

        const group = interaction.options.getSubcommandGroup();
        const subcommand = interaction.options.getSubcommand();

        if (group === "admins" && (await handleAdminSubcommand(interaction, subcommand, ephemeral))) {
            return;
        }

        if (
            group === "moderators" &&
            (await handleModeratorSubcommand(interaction, subcommand, ephemeral))
        ) {
            return;
        }

        if (
            group === "command" &&
            (await handleCommandPolicySubcommand(interaction, subcommand, ephemeral))
        ) {
            return;
        }

        await replyWithAccessMessage(
            interaction,
            "❌ Subcomando não suportado.",
            ephemeral
        );
    },
};

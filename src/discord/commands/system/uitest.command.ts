import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ChatInputCommandInteraction,
    ContainerBuilder,
    MessageFlags,
    SlashCommandBuilder,
    TextDisplayBuilder,
} from "discord.js";
import { getToolDisplay } from "@/tools/registry";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";
import { DISCORD_TOOL_NAMES } from "@/shared/discordTools";
import { isDestructiveTool, isMutatingTool } from "@/tools/registry";

const TOOL_PREVIEW: Record<string, { result: string }> = {
    clear_messages: {
        result: "Apagar 10 mensagem(ns) em <#1255476287733104783>?",
    },
    create_channel: {
        result: "Criar canal de texto #novidades?",
    },
    create_category: {
        result: "Criar categoria 📁 Projetos?",
    },
    delete_channel: {
        result: "Eliminar canal <#1255476287733104783>?",
    },
    create_thread: {
        result: 'Criar thread "Discussão" em <#1255476287733104783>?',
    },
    move_channel: {
        result: "Mover canal <#1255476287733104783> para categoria 🎮 Gaming?",
    },
    manage_member_roles: {
        result: "Alterar cargos de <@123456789012345678>: +Moderador, -Visitante?",
    },
    send_message: {
        result: "Enviar mensagem em <#1255476287733104783>?",
    },
};

// ── Edit this function freely to preview different container styles ──────────
function buildPreviewBlocks(): Array<ContainerBuilder | ActionRowBuilder<ButtonBuilder>> {
    return DISCORD_TOOL_NAMES.filter((toolName) => isMutatingTool(toolName)).flatMap((toolName) => {
        const toolDisplay = getToolDisplay(toolName);
        const preview = TOOL_PREVIEW[toolName] ?? {
            result: "Sem preview definido.",
        };

        const container = new ContainerBuilder()
            .setAccentColor(isDestructiveTool(toolName) ? 0xed4245 : 0xfee75c)
            .addTextDisplayComponents(
                new TextDisplayBuilder().setContent(`### ${toolDisplay.icon} ${toolDisplay.labelPt}`),
                new TextDisplayBuilder().setContent(preview.result),
            );

        const buttons = new ActionRowBuilder<ButtonBuilder>().addComponents(
            new ButtonBuilder()
                .setCustomId(`uitest:approve:${toolName}`)
                .setLabel("Aceitar")
                .setStyle(ButtonStyle.Success),
            new ButtonBuilder()
                .setCustomId(`uitest:deny:${toolName}`)
                .setLabel("Recusar")
                .setStyle(ButtonStyle.Danger),
        );

        return [container, buttons];
    });
}
// ─────────────────────────────────────────────────────────────────────────────

export default {
    data: new SlashCommandBuilder()
        .setName("uitest")
        .setDescription("[dev] Preview the current approval container style")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0)
        .setDMPermission(false),

    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        await SecurityService.initialize();

        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ Apenas administradores podem usar este comando.",
                flags: MessageFlags.Ephemeral,
            });
            return;
        }

        await interaction.reply({
            components: buildPreviewBlocks(),
            flags: MessageFlags.IsComponentsV2,
        });
    },
};

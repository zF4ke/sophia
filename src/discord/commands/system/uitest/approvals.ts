import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    MessageFlags,
    SlashCommandSubcommandBuilder,
    TextDisplayBuilder,
    type ChatInputCommandInteraction,
} from "discord.js";
import { DISCORD_TOOL_NAMES } from "@/shared/discordTools";
import { getToolDisplay, isDestructiveTool, isMutatingTool } from "@/tools/registry";
import type { BotClient } from "@/shared/appTypes";
import { guardUiTestAdmin } from "@/discord/commands/system/uitest/shared";
import type { UiTestSubcommandModule } from "@/discord/commands/system/uitest/types";

const TOOL_PREVIEW: Record<string, { result: string }> = {
    clear_messages: { result: "Apagar 10 mensagem(ns) em <#1255476287733104783>?" },
    create_channel: { result: "Criar canal de texto #novidades?" },
    create_category: { result: "Criar categoria Projetos?" },
    delete_channel: { result: "Eliminar canal <#1255476287733104783>?" },
    create_thread: { result: 'Criar thread "Discussão" em <#1255476287733104783>?' },
    move_channel: { result: "Mover canal <#1255476287733104783> para categoria Gaming?" },
    manage_member_roles: { result: "Alterar cargos de <@123456789012345678>: +Moderador, -Visitante?" },
    send_message: { result: "Enviar mensagem em <#1255476287733104783>?" },
};

function buildApprovalPreviewPayload(toolName: string) {
    const toolDisplay = getToolDisplay(toolName);
    const preview = TOOL_PREVIEW[toolName] ?? { result: "Sem preview definido." };

    const container = new ContainerBuilder()
        .setAccentColor(isDestructiveTool(toolName) ? 0xed4245 : 0xfee75c)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`### ${toolDisplay.icon} ${toolDisplay.labelPt}`),
            new TextDisplayBuilder().setContent(preview.result),
        );

    const buttons = new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder().setCustomId(`uitest:noop:${toolName}:approve`).setLabel("Aceitar").setStyle(ButtonStyle.Success).setDisabled(true),
        new ButtonBuilder().setCustomId(`uitest:noop:${toolName}:deny`).setLabel("Recusar").setStyle(ButtonStyle.Danger).setDisabled(true),
        new ButtonBuilder().setCustomId(`uitest:noop:${toolName}:correct`).setLabel("Recusar e corrigir").setStyle(ButtonStyle.Primary).setDisabled(true),
        new ButtonBuilder().setCustomId(`uitest:noop:${toolName}:stop`).setLabel("Parar execução").setStyle(ButtonStyle.Secondary).setDisabled(true),
    );

    return {
        components: [container, buttons],
        flags: MessageFlags.IsComponentsV2 as const,
    };
}

const uiTestApprovals: UiTestSubcommandModule = {
    name: "approvals",
    description: "Pré-visualizar cartões de aprovação em mensagens separadas",
    register(builder: SlashCommandSubcommandBuilder) {
        return builder
            .setName(this.name)
            .setDescription(this.description);
    },
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        if (!(await guardUiTestAdmin(interaction))) {
            return;
        }

        if (!interaction.channel || !("send" in interaction.channel)) {
            await interaction.reply({
                content: "❌ Este canal não suporta o preview.",
                flags: MessageFlags.Ephemeral,
            });
            return;
        }

        const previews = DISCORD_TOOL_NAMES.filter((toolName) => isMutatingTool(toolName));

        await interaction.reply({
            content: `A enviar ${previews.length} previews de aprovação, um por mensagem.`,
            flags: MessageFlags.Ephemeral,
        });

        for (const toolName of previews) {
            await interaction.channel.send(buildApprovalPreviewPayload(toolName));
        }
    },
};

export default uiTestApprovals;

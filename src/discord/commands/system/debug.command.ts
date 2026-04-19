import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import {
    buildDebugLogsPanel,
    createInitialDebugLogsPanelState,
    rememberDebugLogsPanelState,
} from "@/discord/commands/system/debugLogsPanel";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import { renderDebugControlPanel } from "@/discord/debug/renderDebugControlPanel";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("debug")
        .setDescription("Controlo de debug da Sophia")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0)
        .setDMPermission(false)
        .addSubcommand((sub) =>
            sub.setName("toggle").setDescription("Enable or disable debug mode")
        )
        .addSubcommand((sub) =>
            sub
                .setName("logs")
                .setDescription("Abrir painel de logs do modelo")
        ),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        await SecurityService.initialize();

        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ Apenas administradores podem usar este comando.",
                flags: MessageFlags.Ephemeral,
            });
            return;
        }

        const sub = interaction.options.getSubcommand();

        if (sub === "toggle") {
            await interaction.reply({
                ...renderDebugControlPanel(DebugModeService.isEnabled()),
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        if (sub === "logs") {
            await interaction.deferReply();
            const state = createInitialDebugLogsPanelState(interaction.user.id);
            const reply = await interaction.editReply(buildDebugLogsPanel(state));
            rememberDebugLogsPanelState(reply.id, state);
        }
    },
};

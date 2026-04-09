import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import { renderDebugControlPanel } from "@/discord/debug/renderDebugControlPanel";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("debug")
        .setDescription("Painel global de debug para respostas da Sophia")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0)
        .setDMPermission(false),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        await SecurityService.initialize();

        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ Você não tem permissão para usar este comando.",
            });
            return;
        }

        await interaction.reply({
            ...renderDebugControlPanel(DebugModeService.isEnabled()),
            flags: MessageFlags.IsComponentsV2,
        });
    },
};

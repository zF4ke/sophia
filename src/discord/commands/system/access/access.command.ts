import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { loadAccessPanelData } from "./panelData";
import { buildAccessPanel } from "./panelRenderer";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("access")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0)
        .setDescription("Painel de administração para segurança do bot")
        .setDMPermission(false),
    async execute(interaction: ChatInputCommandInteraction, client: BotClient) {
        await SecurityService.initialize();

        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ Você não tem permissão para usar este comando.",
                flags: MessageFlags.Ephemeral,
            });
            return;
        }

        await interaction.deferReply();

        const payload = await buildAccessPanel(client, await loadAccessPanelData(client), {
            view: "overview",
        });

        await interaction.editReply({
            ...payload,
            flags: MessageFlags.IsComponentsV2,
        });
    },
};

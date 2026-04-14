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
        .setDescription("Debug tools for Sophia")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0)
        .setDMPermission(false)
        .addSubcommand((sub) =>
            sub.setName("toggle").setDescription("Toggle debug mode on/off")
        )
        .addSubcommand((sub) =>
            sub
                .setName("logs")
                .setDescription("Browse model call logs in a navigable panel")
                .addStringOption((opt) =>
                    opt
                        .setName("date")
                        .setDescription("Log date (YYYY-MM-DD). Defaults to today.")
                )
        ),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        await SecurityService.initialize();

        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ You don't have permission to use this command.",
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

            const dateStr = interaction.options.getString("date");
            const preferredFile = dateStr ? `model-output-${dateStr}.jsonl` : null;
            const state = createInitialDebugLogsPanelState(interaction.user.id, preferredFile);
            const reply = await interaction.editReply(buildDebugLogsPanel(state));
            rememberDebugLogsPanelState(reply.id, state);
        }
    },
};

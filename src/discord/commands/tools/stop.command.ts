import { ChatInputCommandInteraction, MessageFlags, SlashCommandBuilder } from "discord.js";
import { ExecutionControl } from "@/runtime/ExecutionControl";

export = {
    data: new SlashCommandBuilder()
        .setName("stop")
        .setDescription("Interrompe os teus pedidos ativos neste canal")
        .addBooleanOption(option => option.setName("ephemeral").setDescription("Mostrar a confirmação só para ti")),
    async execute(interaction: ChatInputCommandInteraction) {
        const count = ExecutionControl.cancelForActor(interaction.user.id, interaction.channelId);
        await interaction.reply({
            flags: interaction.options.getBoolean("ephemeral") ? MessageFlags.Ephemeral : undefined,
            content: count > 0
                ? "Interrupção pedida. Não serão iniciadas novas ações; uma operação em curso pode ainda terminar."
                : "Não tens pedidos ativos neste canal.",
        });
    },
};

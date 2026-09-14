import { ChatInputCommandInteraction, MessageFlags, SlashCommandBuilder } from "discord.js";
import { ExecutionControl } from "@/runtime/ExecutionControl";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { AccessPolicy } from "@/security/AccessPolicy";

export = {
    data: new SlashCommandBuilder()
        .setName("steer")
        .setDescription("Corrige a direção do teu pedido em curso neste canal")
        .addStringOption(option => option.setName("instruction").setDescription("O que deve mudar no pedido em curso")
            .setRequired(true).setMinLength(1).setMaxLength(4000))
        .addStringOption(option => option.setName("task_id").setDescription("Pedido exato, se tens vários em curso"))
        .addBooleanOption(option => option.setName("ephemeral").setDescription("Mostrar a confirmação só para ti")),
    async execute(interaction: ChatInputCommandInteraction) {
        const text = interaction.options.getString("instruction", true).trim();
        const taskId = interaction.options.getString("task_id") ?? undefined;
        let status: "queued" | "not_found" | "ambiguous" | "failed" | "empty" = text ? await ExecutionControl.steerForActor(interaction.user.id, interaction.channelId, text, taskId).catch(() => "failed" as const) : "empty";
        if (status === "not_found" && taskId && interaction.guildId) {
            status = await ExecutionControl.steerAsCollaborator(interaction.user.id, interaction.channelId, text, taskId, async () =>
                await AccessPolicy.decide(interaction.user.id, interaction.guild, "none") === "allow" &&
                await taskStore.canCollaborate(taskId, interaction.user.id, interaction.channelId, interaction.guildId)).catch(() => "failed" as const);
        }
        const replies = {
            failed: "Não consegui guardar a correção. O pedido será pausado para não perder essa alteração.",
            queued: "Correção recebida. Será aplicada na próxima decisão. Uma ação já iniciada pode ainda terminar.",
            not_found: "Não tens um pedido ativo neste canal.",
            ambiguous: "Tens vários pedidos ativos neste canal. Usa /steer com o task_id mostrado em /tasks.",
            empty: "Escreve a correção que queres aplicar.",
        };
        await interaction.reply({ flags: interaction.options.getBoolean("ephemeral") ? MessageFlags.Ephemeral : undefined, content: replies[status] });
    },
};

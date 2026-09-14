import { AttachmentBuilder, MessageFlags, SlashCommandBuilder, type ChatInputCommandInteraction } from "discord.js";
import { scheduleStore } from "@/runtime/scheduling/ScheduleStore";
const labels: Record<string, string> = { active: "Agendado", running: "Em curso", paused: "Pausado", completed: "Concluído", cancelled: "Cancelado" };
export = {
    data: new SlashCommandBuilder().setName("schedules").setDescription("Mostra os teus agendamentos neste local")
        .addStringOption(option => option.setName("schedule_id").setDescription("ID para consultar o histórico e as entregas"))
        .addBooleanOption(option => option.setName("ephemeral").setDescription("Mostrar só para ti")),
    async execute(interaction: ChatInputCommandInteraction) {
        const privateResponse = interaction.options.getBoolean("ephemeral") === true;
        await interaction.deferReply({ flags: privateResponse ? MessageFlags.Ephemeral : undefined });
        const owner = { actorId: interaction.user.id, guildId: interaction.guildId, channelId: interaction.channelId };
        const schedules = await scheduleStore.list(owner);
        const id = interaction.options.getString("schedule_id");
        if (id) {
            const schedule = schedules.find(schedule => schedule.id === id);
            if (!schedule) { await interaction.editReply({ content: "Não encontrei esse agendamento entre os teus agendamentos neste local." }); return; }
            const runs = await scheduleStore.runs(owner, id);
            await interaction.editReply({ content: `${labels[schedule.status] ?? schedule.status} · Revisão ${schedule.revision}`,
                files: [new AttachmentBuilder(Buffer.from(JSON.stringify({ schedule, runs }, null, 2)), { name: "agendamento.json" })], allowedMentions: { parse: [] } });
            return;
        }
        const content = schedules.map(schedule => `${labels[schedule.status] ?? schedule.status} · ${schedule.spec.name}\n${new Date(schedule.nextAt).toLocaleString("pt-PT", { timeZone: schedule.spec.timezone })} · ${schedule.spec.timezone}\nID: ${schedule.id} · Revisão ${schedule.revision}${schedule.reason ? `\n${schedule.reason}` : ""}`).join("\n\n");
        await interaction.editReply(content.length <= 1900
            ? { content: content || "Não tens agendamentos neste local.", allowedMentions: { parse: [] } }
            : { files: [new AttachmentBuilder(Buffer.from(content), { name: "agendamentos.txt" })], allowedMentions: { parse: [] } });
    },
};

import { MessageFlags, type ButtonInteraction } from "discord.js";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { ExecutionControl } from "@/runtime/ExecutionControl";

export async function handleTaskInteraction(interaction: ButtonInteraction): Promise<boolean> {
    const match = /^task:(stop|details):([a-f0-9-]+)$/.exec(interaction.customId);
    if (!match) return false;
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });
    const taskId = match[2];
    const snapshot = await taskStore.snapshot(taskId, interaction.user.id, interaction.channelId, interaction.guildId);
    if (!snapshot) { await interaction.editReply({ content: "Este pedido pertence a outra pessoa ou a outro local." }); return true; }
    if (match[1] === "stop") {
        const stopped = ExecutionControl.cancelTask(taskId, interaction.user.id, interaction.channelId);
        await interaction.editReply({ content: stopped ? "Pedi a interrupção. Uma ação já enviada pode ainda terminar." : "Este pedido já não está a executar." });
        return true;
    }
    const open = snapshot.goals.filter(goal => goal.status === "open" || goal.status === "in_progress").length;
    const uncertain = snapshot.actions.filter(action => action.status === "unknown").length;
    await interaction.editReply({ content: `Pedido: ${taskId}\nObjetivos pendentes: ${open}\nNotas guardadas: ${snapshot.notes.length}\nAções incertas: ${uncertain}\n\nConsulta os ficheiros e o registo completo com /tasks task_id:${taskId}.`, allowedMentions: { parse: [] } });
    return true;
}

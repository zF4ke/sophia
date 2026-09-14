import { randomUUID } from "node:crypto";
import type { AnySelectMenuInteraction, ButtonInteraction } from "discord.js";
import { AccessPolicy } from "@/security/AccessPolicy";
import { ToolExecutor } from "@/runtime/ToolExecutor";
import { createApprovalGate } from "@/discord/approval/ApprovalGate";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { durableApproval } from "@/runtime/tasks/TaskApprovals";

/** Script output is a proposed action, never authority to use the bot client. */
export async function deliverArtifactMessage(
    interaction: ButtonInteraction | AnySelectMenuInteraction,
    outgoing: { channelId: string; content: string },
): Promise<void> {
    const authorize = AccessPolicy.forActor(interaction.user.id, interaction.guild);
    const decision = await authorize("write", "send_message");
    if (decision === "deny") throw new Error("Não tens permissão para enviar mensagens por este cartão.");
    const target = interaction.guild?.channels.cache.get(outgoing.channelId);
    if (!target?.isTextBased()) throw new Error("O destino não está disponível neste servidor.");
    const args = { channel_id: outgoing.channelId, content: outgoing.content };
    const taskId = await taskStore.create({ actorId: interaction.user.id, guildId: interaction.guildId,
        channelId: interaction.channelId, conversationId: `artifact:${interaction.channelId}`, objective: `Send an artifact message to ${outgoing.channelId}` });
    try {
    let approved = false;
    if (decision === "ask") {
        const channel = interaction.channel;
        if (!channel?.isSendable()) throw new Error("Não foi possível pedir aprovação neste canal.");
        const gate = durableApproval({ taskId, actorId: interaction.user.id, channelId: interaction.channelId, guildId: interaction.guildId }, createApprovalGate(channel))!;
        const approval = await gate({
            requestId: randomUUID(), toolName: "send_message", toolArgs: args,
            description: `Enviar para <#${outgoing.channelId}>:\n${outgoing.content}`,
            requesterId: interaction.user.id, sideEffectLevel: "write",
        });
        if (!approval.approved) throw new Error("Envio não aprovado.");
        approved = true;
    }
    const result = await ToolExecutor.execute("send_message", args, {
        taskId,
        guild: interaction.guild, currentChannelId: interaction.channelId,
        actorId: interaction.user.id, question: "Interação com cartão", authorize,
    }, { approved });
    await taskStore.finish(taskId, interaction.user.id, result.uncertainAction ? "paused" : result.record.blocked ? "failed" : "completed", result.record.summary ?? "");
    if (result.record.blocked) throw new Error(result.record.learned);
    } catch (error) {
        await taskStore.finish(taskId, interaction.user.id, "paused", error instanceof Error ? error.message : String(error));
        throw error;
    }
}

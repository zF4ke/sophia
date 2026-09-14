import { MessageFlags, type ChatInputCommandInteraction, type MessageCreateOptions } from "discord.js";
import type { ApprovalTransport } from "./ApprovalGate";
export function interactionApprovalTransport(interaction: ChatInputCommandInteraction): ApprovalTransport | undefined {
    if (!interaction.channel || !('send' in interaction.channel)) return undefined;
    if (interaction.ephemeral === false) {
        const channel = interaction.channel;
        return { send: payload => channel.send({ ...payload, allowedMentions: { parse: [] } }) };
    }
    return { async send(payload: MessageCreateOptions) {
        const message = await interaction.followUp({ ...payload, flags: MessageFlags.IsComponentsV2 | MessageFlags.Ephemeral, allowedMentions: { parse: [] } });
        return { edit: options => interaction.webhook.editMessage(message.id, options) };
    } };
}

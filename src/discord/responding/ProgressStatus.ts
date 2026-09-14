import type { TextBasedChannel, ChatInputCommandInteraction, MessageCreateOptions, MessageEditOptions } from "discord.js";
import { ActionRowBuilder, ButtonBuilder, ButtonStyle, MessageFlags } from "discord.js";
import { ExecutionControl } from "@/runtime/ExecutionControl";

type ProgressMessage = { id?: string; edit(payload: MessageEditOptions): Promise<unknown>; delete(): Promise<unknown> };
type ProgressTransport = { send(payload: MessageCreateOptions): Promise<ProgressMessage> };

const THROTTLE_MS = 1500;
export interface ProgressStatus {
    bindTask(taskId: string): void;
    notify(summary: string): Promise<void>;
    finalize(): Promise<void>;
}

/** One serialized status message; coalesced updates flush even without another call. */
export class ProgressStatusService {
    public static startForInteraction(interaction: ChatInputCommandInteraction): ProgressStatus {
        return this.startForChannel({ async send(payload: MessageCreateOptions) {
            const sent = await interaction.followUp({ ...payload, flags: MessageFlags.Ephemeral });
            return { edit: (next: MessageEditOptions) => interaction.webhook.editMessage(sent.id, next), delete: () => interaction.webhook.deleteMessage(sent.id) };
        } });
    }
    public static startForChannel(channel: TextBasedChannel | ProgressTransport): ProgressStatus {
        let message: ProgressMessage | null = null;
        let pending: string | null = null;
        let lastSummary: string | null = null;
        let lastEditAt = 0;
        let timer: ReturnType<typeof setTimeout> | undefined;
        let chain = Promise.resolve();
        let active = true;
        let recreated = false;
        let taskId: string | undefined;
        const enqueue = (work: () => Promise<void>) => chain = chain.then(work).catch(() => {});
        const schedule = () => {
            if (!active || timer || pending === null) return;
            timer = setTimeout(() => { timer = undefined; void enqueue(flush); }, Math.max(0, THROTTLE_MS - (Date.now() - lastEditAt)));
            timer.unref?.();
        };
        const flush = async () => {
            if (!active || pending === null) return;
            const summary = pending;
            pending = null;
            if (summary === lastSummary) return;
            const payload = { content: `-# ${summary}`, allowedMentions: { parse: [] as never[] },
                ...(taskId ? { components: [new ActionRowBuilder<ButtonBuilder>().addComponents(
                    new ButtonBuilder().setCustomId(`task:stop:${taskId}`).setLabel("Parar").setStyle(ButtonStyle.Secondary),
                    new ButtonBuilder().setCustomId(`task:details:${taskId}`).setLabel("Detalhes").setStyle(ButtonStyle.Secondary))] } : {}) };
            try {
                if (!message) {
                    if (!("send" in channel)) return;
                    message = await channel.send(payload);
                } else await message.edit(payload);
                if (taskId && message.id) ExecutionControl.bindTaskMessage(taskId, message.id);
                lastSummary = summary;
            } catch (error) {
                // Only a confirmed deleted message can be recreated. A timeout
                // does not prove the previous send/edit failed.
                if (!recreated && (error as { code?: number }).code === 10008 && "send" in channel) {
                    recreated = true;
                    message = await channel.send(payload).catch(() => null);
                    if (taskId && message?.id) ExecutionControl.bindTaskMessage(taskId, message.id);
                    if (message) lastSummary = summary;
                } else active = false;
            } finally { lastEditAt = Date.now(); schedule(); }
        };
        return {
            bindTask(id) { taskId = id; },
            async notify(summary) {
                if (!active) return;
                const cleaned = summary.replace(/[\r\n]+/g, " ").trim().slice(0, 350);
                if (!cleaned || cleaned === lastSummary && pending === null) return;
                pending = cleaned;
                if (!message || Date.now() - lastEditAt >= THROTTLE_MS) await enqueue(flush);
                else schedule();
            },
            async finalize() {
                active = false;
                pending = null;
                if (timer) clearTimeout(timer);
                await chain;
                if (message) await message.delete().catch(() => {});
                message = null;
            },
        };
    }
}

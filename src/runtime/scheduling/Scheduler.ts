import { AttachmentBuilder, PermissionFlagsBits, type Client } from "discord.js";
import { scheduleStore, type ScheduleRecord } from "./ScheduleStore";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";
import { BackgroundLoop } from "@/app/BackgroundLoop";
import { SettingsService } from "@/app/SettingsService";
import { AccessPolicy } from "@/security/AccessPolicy";
import { Runtime } from "../Runtime";
import { createApprovalGate, createBatchApprovalGate } from "@/discord/approval/ApprovalGate";
import { ConversationAdapter } from "@/discord/conversation/ConversationAdapter";
import type { TurnInput } from "../contracts";
import { taskStore } from "../tasks/TaskStore";

export class Scheduler {
    private static readonly loop = new BackgroundLoop();
    private static busy = false;
    static start(client: Client) {
        this.loop.start(() => this.tick(client), () => SettingsService.load().scheduling.pollIntervalMs, error => console.error("[Scheduler]", error));
    }
    static stop() { this.loop.stop(); }
    static async tick(client: Client, execute = this.execute.bind(this)) {
        if (this.busy || !ActiveRequestTracker.isIdle() || !SettingsService.load().scheduling.enabled) return;
        this.busy = true;
        const release = ActiveRequestTracker.begin();
        try {
            const job = await scheduleStore.claim();
            if (!job) return;
            try { await execute(client, job.schedule, job.runId); }
            catch (error) { await scheduleStore.finish(job.runId, job.schedule, { error: error instanceof Error ? error.message : String(error) }); }
        } finally { release(); this.busy = false; }
    }
    static async execute(client: Client, schedule: ScheduleRecord, runId: string) {
        const { owner, spec } = schedule;
        const guild = owner.guildId ? await client.guilds.fetch(owner.guildId) : null;
        const channel = await client.channels.fetch(owner.channelId!);
        if (!channel?.isSendable()) throw new Error("Scheduled destination is unavailable.");
        if (guild && (!('guildId' in channel) || channel.guildId !== guild.id)) throw new Error("Scheduled destination no longer belongs to this guild.");
        if (!guild && (!channel.isDMBased() || !('recipientId' in channel) || channel.recipientId !== owner.actorId)) throw new Error("Scheduled DM destination does not belong to its owner.");
        const authorize: NonNullable<TurnInput["authorize"]> = async (effect, tool) => {
            if (!SettingsService.load().scheduling.enabled || !await scheduleStore.current(schedule)) return "deny";
            const decision = await AccessPolicy.decide(owner.actorId, guild, effect, tool);
            if (decision === "deny") return decision;
            if (guild) {
                const member = await guild.members.fetch({ user: owner.actorId, force: true }).catch(() => null);
                if (!member || !('permissionsFor' in channel) || !channel.permissionsFor(member)?.has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.ReadMessageHistory])) return "deny";
            }
            return decision;
        };
        if (await authorize("none") === "deny") throw new Error("The schedule owner's access is unavailable or revoked.");
        const user = await client.users.fetch(owner.actorId);
        const previousRun = (await scheduleStore.runs(owner, schedule.id)).find(run => run.status === "completed" && run.task_id);
        const priorScheduledResult = previousRun ? await taskStore.completedResult(String(previousRun.task_id), owner.actorId, owner.channelId!, owner.guildId) : null;
        const input: TurnInput = {
            user, guild, currentChannelId: owner.channelId, requesterDisplayName: user.displayName || user.username,
            question: spec.prompt, trigger: "scheduled", authorize, allowDreaming: false,
            notificationPolicy: spec.notificationPolicy ?? "always", priorScheduledResult,
            conversation: { key: `schedule:${schedule.id}`, kind: "channel", trigger: "scheduled" },
            approvalGate: createApprovalGate(channel), batchApprovalGate: createBatchApprovalGate(channel),
        };
        const result = await Runtime.answer(input);
        if (!result.taskId) throw new Error("Scheduled work did not produce a durable task.");
        if (await authorize("write", "send_message") === "deny") throw new Error("Scheduled result delivery is no longer authorized.");
        if (spec.notificationPolicy === "conditional" && result.notify === false && result.outcome === "completed") {
            await scheduleStore.finish(runId, schedule, { taskId: result.taskId });
            return;
        }
        // The approved schedule authorizes this result delivery only. Other
        // mutations inside the task still pass through the usual approval gate.
        await scheduleStore.prepareDelivery(runId, schedule, result.taskId);
        let delivered = false;
        try {
            if (!await scheduleStore.current(schedule)) throw new Error("Schedule changed before delivery.");
            if (await authorize("write", "send_message") === "deny") throw new Error("Scheduled result delivery was revoked before dispatch.");
            const fullText = `${spec.name}\n${result.answer}\n\nPedido: ${result.taskId}`;
            if (Buffer.byteLength(fullText) > 8 * 1024 * 1024) throw new Error("Scheduled result exceeds the delivery attachment limit; inspect the task.");
            const message = await channel.send(fullText.length <= 1900
                ? { content: `<@${owner.actorId}> ${fullText}`, allowedMentions: { users: [owner.actorId] } }
                : { content: `<@${owner.actorId}> ${spec.name}\nPedido: ${result.taskId}`, files: [new AttachmentBuilder(Buffer.from(fullText), { name: "resultado.txt" })], allowedMentions: { users: [owner.actorId] } });
            delivered = true;
            await scheduleStore.finish(runId, schedule, { taskId: result.taskId, messageId: message.id, ...(result.outcome !== "completed" ? { error: `Task ${result.taskId} ${result.outcome ?? "paused"}` } : {}) });
            await ConversationAdapter.bindResponseMessages({ input, result, sentMessages: [message] });
        } catch (error) {
            await scheduleStore.finish(runId, schedule, { taskId: result.taskId, error: `${delivered ? "Delivery receipt or response binding failed" : "Delivery failed or outcome unknown"}: ${error instanceof Error ? error.message : String(error)}` });
            throw error;
        }
    }
}

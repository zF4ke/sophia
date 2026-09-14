import type { Guild } from "discord.js";
import { assertReadableChannels } from "@/security/SourceAccess";
import { taskStore } from "./TaskStore";
import type { CapabilityContext } from "@/tools/types";

type Observation = { resolution: "applied" | "not_applied" | "unknown"; detail: string };
type Action = NonNullable<Awaited<ReturnType<typeof taskStore.unresolvedAction>>>;
async function absentOn<T>(read: () => Promise<T>, code: number): Promise<T | null> {
    try { return await read(); } catch (error) {
        if (error && typeof error === "object" && "code" in error && Number(error.code) === code) return null;
        throw error;
    }
}
async function inspect(guild: Guild, actorId: string, action: Action): Promise<Observation> {
    const args = action.args;
    if (action.tool === "delete_channel") {
        const id = String(args.channel_id ?? "");
        if (!id) throw new Error("Receipt has no channel ID.");
        const channel = await absentOn(() => guild.channels.fetch(id, { force: true }), 10003);
        return { resolution: channel ? "not_applied" : "applied", detail: `Channel ${id} is ${channel ? "still present" : "absent"} in the current Discord observation.` };
    }
    if (action.tool === "delete_role") {
        const id = String(args.role_id ?? "");
        if (!id) throw new Error("Receipt has no role ID.");
        const role = await absentOn(() => guild.roles.fetch(id, { force: true }), 10011);
        return { resolution: role ? "not_applied" : "applied", detail: `Role ${id} is ${role ? "still present" : "absent"} in the current Discord observation.` };
    }
    if (action.tool === "delete_messages" || action.tool === "edit_message" || action.tool === "send_message") {
        const channelId = String(args.channel_id ?? "");
        await assertReadableChannels(guild, actorId, [channelId]);
        const channel = await guild.channels.fetch(channelId, { force: true });
        if (!channel || !channel.isTextBased() || !("messages" in channel)) throw new Error("Message channel is unavailable.");
        const result = action.result as { data?: { messageId?: unknown } } | null;
        const ids = action.tool === "delete_messages" && Array.isArray(args.message_ids) ? args.message_ids.map(String)
            : [String(action.tool === "send_message" ? result?.data?.messageId ?? "" : args.message_id ?? "")];
        if (!ids.length || ids.some(id => !id)) return { resolution: "unknown", detail: "No exact message ID was preserved; matching by text would not prove which action produced a message." };
        const messages = await Promise.all(ids.map(id => absentOn(() => channel.messages.fetch({ message: id, force: true }), 10008)));
        if (action.tool === "delete_messages") {
            const missing = ids.filter((_, index) => !messages[index]);
            return { resolution: missing.length === ids.length ? "applied" : missing.length === 0 ? "not_applied" : "unknown", detail: `${missing.length}/${ids.length} exact target messages are absent in channel ${channelId}. ${missing.length > 0 && missing.length < ids.length ? "The operation is partial; do not replay the whole batch." : ""}` };
        }
        const message = messages[0];
        const matches = message && message.author.id === guild.client.user?.id && message.content === args.content;
        return { resolution: matches ? "applied" : "unknown", detail: `Message ${ids[0]} ${matches ? "currently matches the requested bot-authored content" : "does not establish the requested result; it may have changed since dispatch"} in channel ${channelId}.` };
    }
    return { resolution: "unknown", detail: `No reliable automatic postcondition verifier is available for ${action.tool}. Inspect the effect before recording a manual resolution.` };
}

export class ActionVerifier {
    static async verify(context: CapabilityContext, taskId: string, actionId: string) {
        if (!context.actorId || !context.guild || !context.authorize || await context.authorize("none") === "deny") throw new Error("Verification requires current authenticated guild access.");
        const action = await taskStore.unresolvedAction(taskId, actionId, context.actorId, context.currentChannelId ?? "", context.guild.id);
        if (!action) throw new Error("Unknown action is not owned by this requester in a paused task here.");
        const observation = await inspect(context.guild, context.actorId, action);
        context.execution?.checkpoint();
        if (await context.authorize("none") === "deny") throw new Error("Verification authority changed.");
        if (observation.resolution !== "unknown") await taskStore.resolveAction({ taskId, actionId, actorId: context.actorId, channelId: context.currentChannelId ?? "", guildId: context.guild.id,
            resolution: observation.resolution, verification: observation.detail, verificationKind: "discord_observation" });
        return { taskId, actionId, tool: action.tool, ...observation, resumed: false };
    }
}

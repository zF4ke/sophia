import type { CapabilityContext } from "@/tools/types";
import { DiscordHistoryReader } from "./DiscordHistoryReader";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { assertReadableChannels } from "@/security/SourceAccess";
import { taskStore } from "@/runtime/tasks/TaskStore";

export async function resolveDiscordAttachment(context: CapabilityContext, attachmentId: string, messageUrl?: string) {
    const supplied = context.attachments?.find(file => file.id === attachmentId);
    const source = messageUrl ?? await DiscordMemoryService.findAttachmentMessageAsync(attachmentId);
    if (!source) {
        if (!supplied) throw new Error("Attachment is not available in this turn or index. Provide the original Discord message URL with this attachment ID.");
        return { attachment: supplied, sourceMessageIds: [] as string[], sourceChannelIds: [] as string[] };
    }
    const url = new URL(source);
    const match = /^\/channels\/(\d+|@me)\/(\d+)\/(\d+)$/.exec(url.pathname);
    if (url.protocol !== "https:" || !["discord.com", "canary.discord.com", "ptb.discord.com"].includes(url.hostname) || url.username || url.password || url.port || !match) throw new Error("Use the original Discord message link, not the CDN URL.");
    const [, guildId, channelId, messageId] = match;
    context.execution?.checkpoint();
    if (guildId === "@me") {
        if (context.guild || context.currentChannelId !== channelId) throw new Error("Attachment is outside this DM conversation.");
    } else await assertReadableChannels(context.guild, context.actorId, [channelId], { client: context.client, privateResponse: context.privateResponse, destinationChannelId: context.currentChannelId });
    const channel = await (context.client ?? context.guild?.client)?.channels.fetch(channelId);
    if (!channel?.isTextBased() || !("messages" in channel)) throw new Error("The attachment's original channel is unavailable.");
    const message = await DiscordHistoryReader.message(channel, messageId);
    const attachment = message?.attachments.get(attachmentId);
    if (!attachment) throw new Error("This attachment is no longer on the original message.");
    if (context.taskId && context.requestId && context.actorId) await taskStore.recordEvidenceSources({ taskId: context.taskId, requestId: context.requestId, actorId: context.actorId, channelId: context.currentChannelId ?? "", guildId: context.guild?.id ?? null }, [{ messageId, channelId, guildId: guildId === "@me" ? null : guildId, sourceUrl: (await DiscordMemoryService.getStoredMessageAsync(messageId))?.jumpLink ?? source }]);
    context.execution?.watchSources([messageId]);
    return { attachment: { id: attachment.id, name: attachment.name, url: attachment.url, contentType: attachment.contentType, size: attachment.size }, sourceMessageIds: [messageId], sourceChannelIds: guildId === "@me" ? [] : [channelId] };
}

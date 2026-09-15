import type { TurnInput } from "./contracts";
import type { SourceChange } from "./ExecutionControl";
import { sourceReference } from "@/shared/sourceReference";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { taskStore } from "./tasks/TaskStore";
import { assertReadableChannels } from "@/security/SourceAccess";
import { DiscordHistoryReader } from "@/discord/live/DiscordHistoryReader";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

type Source = { messageId: string; channelId: string | null; guildId: string | null; sourceUrl?: string };
/** Refresh only affected messages. A failed read becomes a gap, not a task failure. */
export async function refreshSources(input: TurnInput, sources: Source[], changes: SourceChange[]) {
    const retained: Source[] = [];
    const updates: Array<{ messageId: string; url: string; content?: string; author?: string; createdAt?: string; attachments?: unknown; limitation?: string }> = [];
    const audience = { client: input.user.client, privateResponse: input.responseVisibility === "private" || !input.guild, destinationChannelId: input.currentChannelId };
    const access = new Map<string, Promise<void>>();
    for (const source of sources) {
        input.execution?.checkpoint();
        const url = sourceReference(source).split("#")[0];
        let changed = changes.some(change => change.messageId === source.messageId) || await knowledgeStore.isSourceInvalid(sourceReference(source));
        try {
            const parsed = new URL(url);
            const sourceChannel = source.channelId ?? parsed.pathname.split("/")[3];
            const sourceGuild = source.guildId ?? parsed.pathname.split("/")[2];
            if (sourceGuild !== "@me") {
                if (!access.has(sourceChannel)) access.set(sourceChannel, assertReadableChannels(input.guild, input.user.id, [sourceChannel], audience));
                await access.get(sourceChannel);
            }
            if (await taskStore.hasDeletedSources([source.messageId])) {
                updates.push({ messageId: source.messageId, url, limitation: "Message was deleted." });
                continue;
            }
            const current = await taskStore.currentMessageSource(source.messageId);
            changed ||= Boolean(current && current !== sourceReference(source));
            if (!changed) { retained.push(source); continue; }
            const channel = await input.user.client.channels.fetch(sourceChannel);
            if (!channel?.isTextBased() || !("messages" in channel)) throw new Error("Channel history is unavailable.");
            const message = await DiscordHistoryReader.message(channel, source.messageId);
            if (!message) { updates.push({ messageId: source.messageId, url, limitation: "Transient task status, excluded from evidence." }); continue; }
            const stored = await DiscordMemoryService.getStoredMessageAsync(source.messageId);
            if (!stored) throw new Error("Message is no longer available after refresh.");
            const sourceUrl = stored.jumpLink;
            retained.push({ messageId: source.messageId, channelId: sourceChannel, guildId: sourceGuild === "@me" ? null : sourceGuild, sourceUrl });
            updates.push({ messageId: source.messageId, url, content: stored.content, author: stored.authorName, createdAt: new Date(stored.createdTimestamp).toISOString(), attachments: JSON.parse(stored.attachmentsJson) });
        } catch (error) {
            updates.push({ messageId: source.messageId, url, limitation: error instanceof Error ? error.message : "Could not refresh this message." });
        }
    }
    input.execution?.checkpoint();
    return { retained, updates };
}

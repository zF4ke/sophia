import { sourceReference } from "@/shared/sourceReference";
import { assertReadableChannels } from "./SourceAccess";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { taskStore } from "@/runtime/tasks/TaskStore";
import type { CapabilityContext } from "@/tools/types";

export function discordSource(source: string) {
    try {
        const url = new URL(source);
        if (!(["discord.com", "www.discord.com", "canary.discord.com", "ptb.discord.com"].includes(url.hostname))) return null;
        const match = /^\/channels\/([^/]+)\/([^/]+)\/([^/]+)$/.exec(url.pathname);
        return match ? { guildId: match[1] === "@me" ? null : match[1], channelId: match[2], messageId: match[3], sourceUrl: source } : null;
    } catch { return null; }
}

export async function captureDerivedSources(context: CapabilityContext, explicit: string[] = []): Promise<string[]> {
    const current = context.requestId ? await taskStore.requestSources(context.requestId) ?? [] : [];
    return [...new Set([...explicit, ...current.map(source => sourceReference(source))])];
}

export async function assertDerivedSources(context: CapabilityContext, sources: string[], sharedGuild = false): Promise<void> {
    for (const source of sources) if (await knowledgeStore.isSourceInvalid(source)) throw new Error("A source of this saved content was deleted.");
    const references = sources.flatMap(source => { const parsed = discordSource(source); return parsed ? [parsed] : []; });
    const privateResponse = !sharedGuild && (context.privateResponse || !context.guild);
    if (references.some(source => !source.guildId) && !privateResponse) throw new Error("Private message sources cannot be disclosed to this audience.");
    await assertReadableChannels(context.guild, context.actorId, references.flatMap(source => source.guildId ? [source.channelId] : []), {
        client: context.client, privateResponse, destinationChannelId: sharedGuild ? "guild-wide-audience" : context.currentChannelId,
    });
    context.execution?.watchSources(references.map(source => source.messageId));
    if (context.taskId && context.actorId && context.requestId && references.length) await taskStore.recordEvidenceSources({ taskId: context.taskId, actorId: context.actorId, requestId: context.requestId, channelId: context.currentChannelId ?? "", guildId: context.guild?.id ?? null }, references);
    for (const source of sources) if (await knowledgeStore.isSourceInvalid(source)) throw new Error("A saved source changed during access verification.");
}

export async function readableDerived<T>(context: CapabilityContext, items: T[], sources: (item: T) => string[], privateOnly: (item: T) => boolean = () => false): Promise<T[]> {
    const readable: T[] = [];
    for (const item of items) {
        if (privateOnly(item) && context.guild && !context.privateResponse) continue;
        try { await assertDerivedSources(context, sources(item)); readable.push(item); } catch { /* Unavailable content does not enter model context. */ }
    }
    return readable;
}

/** A private model context does not make a channel publication private. */
export async function assertPublicationAudience(context: CapabilityContext, destinationChannelId: string, explicitSources: string[] = []): Promise<void> {
    const sources = await captureDerivedSources(context, explicitSources);
    for (const source of sources) if (await knowledgeStore.isSourceInvalid(source)) throw new Error("A publication source was deleted.");
    const references = sources.flatMap(source => { const parsed = discordSource(source); return parsed ? [parsed] : []; });
    if (context.guild && references.some(source => !source.guildId)) throw new Error("Private message sources cannot be published to a guild channel.");
    await assertReadableChannels(context.guild, context.actorId, references.flatMap(source => source.guildId ? [source.channelId] : []), {
        client: context.client, privateResponse: !context.guild, destinationChannelId,
    });
    for (const source of sources) if (await knowledgeStore.isSourceInvalid(source)) throw new Error("A publication source changed during access verification.");
}

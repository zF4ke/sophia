import type { Guild } from "discord.js";
import { assertReadableChannels, type SourceAudience } from "@/security/SourceAccess";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import type { ToolInvocationRecord } from "./contracts";

/** Revalidate persisted evidence before replay or private export. */
export async function readableToolRecords(records: ToolInvocationRecord[], guild: Guild | null, actorId: string, audience: SourceAudience = {}): Promise<ToolInvocationRecord[]> {
    // Tools can request evidence review themselves. Resolve the completed
    // catalog at call time rather than during tool-module initialization.
    const { getToolStrategy } = await import("@/tools/registry");
    const access = new Map<string, Promise<boolean>>();
    const canRead = (channelId: string) => {
        if (!access.has(channelId)) access.set(channelId, assertReadableChannels(guild, actorId, [channelId], audience).then(() => true, () => false));
        return access.get(channelId)!;
    };
    const result: ToolInvocationRecord[] = [];
    for (const record of records) {
        if (!record.output) { result.push(record); continue; }
        const evidence = getToolStrategy(record.tool).extractEvidence(record.output).filter(item => item.messageId);
        let available = true;
        for (const item of evidence) {
            if ((item.channelId && !await canRead(item.channelId)) || (item.jumpLink && await knowledgeStore.isSourceInvalid(item.jumpLink))) { available = false; break; }
        }
        result.push(available ? record : { tool: record.tool, arguments: {}, summary: "Prior source is no longer available.", learned: "Retrieve current evidence before relying on this result.", output: { tool: record.tool, summary: "Source unavailable.", data: null, errorMessage: "Source access changed." }, confidenceImproved: false, durationMs: record.durationMs, blocked: true, sourceUnavailable: true });
    }
    return result;
}

import { z } from "zod";
import { ChannelType } from "discord.js";
import { T } from "@/shared/discordTools";
import { DiscordBackfillCrawler } from "@/discord/live/DiscordBackfillCrawler";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { ToolDefinition } from "./types";

function resolveTargets(context: { guild: import("discord.js").Guild | null; currentChannelId?: string | null }, channelIds: string[]): import("discord.js").TextChannel[] {
    const guild = context.guild;
    if (!guild) return [];
    const targets: import("discord.js").TextChannel[] = [];
    const wanted = channelIds.length ? channelIds : context.currentChannelId ? [context.currentChannelId] : [];
    for (const id of wanted) {
        const ch = guild.channels.cache.get(id);
        if (ch && (ch.type === ChannelType.GuildText || ch.type === ChannelType.GuildAnnouncement)) {
            targets.push(ch as import("discord.js").TextChannel);
        }
    }
    return targets;
}

const params = {
    type: "object",
    properties: {
        channel_ids: {
            type: "array",
            description: "Channel IDs to index. Defaults to the current channel.",
            items: { type: "string" },
        },
        mode: {
            type: "string",
            description: "'refresh' (default) = fetch newest messages to close offline gaps (fast). 'deep' = queue full history backfill via the background crawler (slow, walks to the very beginning).",
        },
        limit: {
            type: "number",
            description: "For refresh mode: max messages to fetch (100-2000, default 500).",
        },
    },
    required: [],
} as const;

export const indexChannelTool: ToolDefinition = {
    name: T.index_channel,

    catalog: {
        effect: "read",
        description: "Update the local message index for channels: fetch fresh messages now or queue a deep history backfill.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Index Discord channels into local memory. Use when the index looks stale relative to the current date (see Index Freshness in context) or when retrieval returns old/partial results. mode='refresh' fetches the newest messages now (fast). mode='deep' queues a full history backfill in the background crawler (replaces the old /index command).",
        parameters: params,
    },

    capability: {
        description: "Index channels (refresh or deep backfill).",
        inputSchema: z.object({
            channel_ids: z.array(z.string()).optional(),
            mode: z.string().optional(),
            limit: z.number().min(100).max(2000).optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: [],
        postconditions: ["channel index updated or backfill queued"],
        async run(context, args) {
            const guild = context.guild;
            if (!guild) return { tool: T.index_channel, summary: "No guild context.", data: null, errorMessage: "Indexing requires a guild." };

            const rawIds = Array.isArray(args.channel_ids) ? (args.channel_ids as string[]).map(String) : [];
            const mode = String(args.mode || "refresh").toLowerCase();
            const limit = Math.max(100, Math.min(2000, Number(args.limit) || 500));
            const targets = resolveTargets(context, rawIds);

            if (!targets.length) {
                return { tool: T.index_channel, summary: "No indexable channels resolved.", data: null, errorMessage: "Provide valid text channel_ids (or call from a guild text channel)." };
            }

            const results: Array<{ channelId: string; channelMention: string; mode: string; ingested: number; queued: boolean; lastIndexedTimestamp: number | null; note?: string }> = [];

            for (const channel of targets) {
                if (mode === "deep") {
                    await DiscordBackfillCrawler.enqueue(channel.id, { reason: "agent_deep", priority: 10, guildId: guild.id });
                    results.push({ channelId: channel.id, channelMention: `<#${channel.id}>`, mode: "deep", ingested: 0, queued: true, lastIndexedTimestamp: null, note: "Full backfill queued in background crawler" });
                    continue;
                }
                try {
                    const r = await DiscordBackfillCrawler.refreshChannel(channel, limit);
                    results.push({
                        channelId: channel.id,
                        channelMention: `<#${channel.id}>`,
                        mode: "refresh",
                        ingested: r.ingested,
                        queued: false,
                        lastIndexedTimestamp: r.lastIndexedTimestamp,
                        note: r.hitKnown ? "Reached already-indexed messages" : r.ingested === 0 ? "Nothing new" : "Sweep limit reached — consider mode:'deep' for full backfill",
                    });
                } catch (e) {
                    const msg = e instanceof Error ? e.message : String(e);
                    results.push({ channelId: channel.id, channelMention: `<#${channel.id}>`, mode: "refresh", ingested: 0, queued: false, lastIndexedTimestamp: null, note: `Error: ${msg}` });
                }
            }

            const totalIngested = results.reduce((a, r) => a + r.ingested, 0);
            const summary = results
                .map((r) => `${r.channelMention} [${r.mode}] ingested=${r.ingested}${r.queued ? " queued=deep-backfill" : ""} — ${r.note ?? ""}`)
                .join("\n");
            return {
                tool: T.index_channel,
                summary: `Indexed ${targets.length} channel(s), ${totalIngested} new messages:\n${summary}`,
                data: { results, totalIngested },
            };
        },
    },

    strategy: { extractEvidence() { return []; } },
    display: { icon: "📥", labelPt: "Indexar canal" },
};

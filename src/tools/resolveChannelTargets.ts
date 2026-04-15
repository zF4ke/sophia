import { z } from "zod";
import { T } from "@/shared/discordTools";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import type { DiscordToolResult, ResolvedChannelTarget } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";
import {
    asGuildStructureEntries,
    isCategoryStructureEntry,
    type GuildStructurePayload,
} from "./listGuildStructure";

const parameters = {
    type: "object",
    properties: {
        targets: {
            type: "array",
            items: { type: "string" },
            description:
                "One or more channel/category names, IDs, or mentions to resolve. Pass all targets in a single call instead of calling this tool multiple times.",
        },
    },
    required: ["targets"],
} as const;

export const resolveChannelTargetsTool: ToolDefinition = {
    name: T.resolve_channel_targets,

    catalog: {
        effect: "read",
        description:
            "Resolve channel or category references in the current guild, including exact ids and category expansion.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Resolve one or more channel or category references in the current guild. Pass an array of channel names, IDs, or mentions. If a category is matched, expands it to all child message channels. Returns resolved channel IDs you can use with retrieve_messages. Always pass all targets in a single call — do not call this tool once per channel.",
        parameters,
    },

    capability: {
        description:
            "Resolve channel or category references in the current guild, including exact ids and category expansion.",
        inputSchema: z.object({
            targets: z.array(z.string()).describe("Channel/category names, IDs, or mentions."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: [
            "returns resolved message-channel targets for the current guild",
        ],
        async run(context, args) {
            const rawTargets = Array.isArray(args.targets) ? args.targets.map(String) : [String(args.targets || context.question)];
            const results =
                await DiscordGuildDiscoveryService.resolveChannelTargetsBatch(
                    context.guild,
                    rawTargets,
                    context.currentChannelId,
                );
            const totalEntries = results.reduce((sum, r) => sum + r.entries.length, 0);
            const totalIds = results.reduce((sum, r) => sum + r.resolvedIds.length, 0);
            return {
                tool: T.resolve_channel_targets,
                summary: totalEntries
                    ? `Resolved ${totalEntries} guild structure target(s) with ${totalIds} message channel id(s) across ${rawTargets.length} queries.`
                    : "No matching channel or category targets were resolved.",
                data: { results },
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data) return [];

            const results = extractResults(run.data);

            return results.flatMap((result) => {
                const entries = asGuildStructureEntries(result.entries);
                return entries.map((entry) => ({
                    tool: T.resolve_channel_targets,
                    summary: run.summary,
                    content: isCategoryStructureEntry(entry)
                        ? `Category ${entry.name} resolved with ${Array.isArray(result.resolvedIds) ? result.resolvedIds.length : 0} visible message channels.`
                        : `Channel #${entry.name}${entry.parentCategoryName ? ` in category ${entry.parentCategoryName}` : ""}${entry.channelTopic ? `. Topic: ${entry.channelTopic.replace(/\s+/g, " ").trim().slice(0, 120)}` : ""}.`,
                    evidenceRole: "discovery_only" as const,
                    strength: "metadata" as const,
                    sourceOrigin: "none" as const,
                    channelId: entry.id,
                    channelName: entry.name,
                }));
            });
        },

        extractResolvedChannel(
            run: DiscordToolResult,
        ): ResolvedChannelTarget | null {
            if (!run.data || typeof run.data !== "object") return null;

            const results = extractResults(run.data);
            if (!results.length) return null;

            // Merge all results into a single composite target
            const allEntries: import("@/shared/appTypes").GuildStructureEntry[] = [];
            const allIds = new Set<string>();
            const queries: string[] = [];
            let anyExactId = false;
            let bestConfidence: ResolvedChannelTarget["confidence"] = "low";

            const confidenceRank = { exact: 4, high: 3, medium: 2, low: 1 };

            for (const result of results) {
                const item = result as GuildStructurePayload & Record<string, unknown>;
                const entries = asGuildStructureEntries(item.entries);
                const resolvedIds = Array.isArray(item.resolvedIds)
                    ? (item.resolvedIds as unknown[]).map(String)
                    : [];
                const query = typeof item.query === "string" && item.query.trim() ? item.query.trim() : "";

                if (query) queries.push(query);
                for (const e of entries) allEntries.push(e);
                for (const id of resolvedIds) allIds.add(id);
                if (item.exactIdMatch) anyExactId = true;

                const conf = (item.confidence === "exact" || item.confidence === "high" || item.confidence === "medium" || item.confidence === "low")
                    ? item.confidence
                    : resolvedIds.length ? "high" : "low";
                if (confidenceRank[conf] > confidenceRank[bestConfidence]) bestConfidence = conf;
            }

            if (!queries.length && !allEntries.length && !allIds.size) return null;

            return {
                query: queries.join(", "),
                resolvedIds: [...allIds],
                entries: allEntries,
                exactIdMatch: anyExactId,
                confidence: bestConfidence,
            };
        },
    },

    display: { icon: "🎯", labelPt: "Encontrar canais" },
};

function extractResults(data: unknown): Array<Record<string, unknown>> {
    if (!data || typeof data !== "object") return [];
    const obj = data as Record<string, unknown>;
    // Batched format: { results: [...] }
    if (Array.isArray(obj.results)) {
        return obj.results as Array<Record<string, unknown>>;
    }
    // Legacy single-result format (in case old tool runs are replayed)
    if (obj.query !== undefined || obj.entries !== undefined) {
        return [obj as Record<string, unknown>];
    }
    return [];
}

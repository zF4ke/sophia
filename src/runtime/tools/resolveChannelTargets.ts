import type { EvidenceItem } from "@/runtime/contracts";
import type { DiscordToolResult, ResolvedChannelTarget } from "@/shared/appTypes";
import { DISCORD_TOOL_EVIDENCE_ROLES } from "@/shared/discordTools";
import { asGuildStructureEntries, isCategoryStructureEntry } from "./guildStructure";
import type { GuildStructurePayload, ToolStrategy } from "./types";

export const resolveChannelTargetsStrategy: ToolStrategy = {
    id: "resolve_channel_targets",

    extractEvidence(run: DiscordToolResult): EvidenceItem[] {
        if (!run.data) {
            return [];
        }

        const item = run.data as { entries?: Array<Record<string, unknown>>; resolvedIds?: string[] };
        const entries = asGuildStructureEntries(item.entries);

        return entries.map((entry) => ({
            tool: "resolve_channel_targets" as const,
            summary: run.summary,
            content: isCategoryStructureEntry(entry)
                ? `Category ${entry.name} resolved with ${Array.isArray(item.resolvedIds) ? item.resolvedIds.length : 0} visible message channels.`
                : `Channel #${entry.name}${entry.parentCategoryName ? ` in category ${entry.parentCategoryName}` : ""}.`,
            evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.resolve_channel_targets,
            strength: "metadata" as const,
            sourceOrigin: "none" as const,
            channelId: entry.id,
            channelName: entry.name,
        }));
    },

    extractResolvedChannel(run: DiscordToolResult): ResolvedChannelTarget | null {
        if (!run.data || typeof run.data !== "object") {
            return null;
        }

        const item = run.data as GuildStructurePayload & Record<string, unknown>;
        const entries = asGuildStructureEntries(item.entries);
        const resolvedIds = Array.isArray(item.resolvedIds)
            ? (item.resolvedIds as unknown[]).map(String)
            : [];
        const query =
            typeof item.query === "string" && item.query.trim() ? item.query.trim() : "";

        if (!query && !entries.length && !resolvedIds.length) {
            return null;
        }

        return {
            query,
            resolvedIds,
            entries,
            exactIdMatch: Boolean(item.exactIdMatch),
            confidence:
                item.confidence === "exact" ||
                item.confidence === "high" ||
                item.confidence === "medium" ||
                item.confidence === "low"
                    ? item.confidence
                    : resolvedIds.length
                      ? "high"
                      : "low",
        };
    },
};

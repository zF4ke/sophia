import type { EvidenceItem } from "@/runtime/contracts";
import type { DiscordToolResult, ResolvedChannelTarget } from "@/shared/appTypes";
import { asGuildStructureEntries, buildStructureEvidenceItems } from "./guildStructure";
import type { GuildStructurePayload, ToolStrategy } from "./types";

export const listGuildStructureStrategy: ToolStrategy = {
    id: "list_guild_structure",

    extractEvidence(run: DiscordToolResult): EvidenceItem[] {
        if (!run.data) {
            return [];
        }

        return buildStructureEvidenceItems(run.data as GuildStructurePayload, run.summary);
    },

    extractResolvedChannel(run: DiscordToolResult): ResolvedChannelTarget | null {
        if (!run.data || typeof run.data !== "object") {
            return null;
        }

        const item = run.data as GuildStructurePayload & Record<string, unknown>;
        const entries = asGuildStructureEntries(item.focusedEntries);
        const resolvedIds = Array.isArray(item.focusedResolvedIds)
            ? (item.focusedResolvedIds as unknown[]).map(String)
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

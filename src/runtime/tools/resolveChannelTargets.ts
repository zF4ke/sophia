import type { EvidenceItem, ToolArguments } from "@/runtime/contracts";
import type { DiscordToolResult, ResolvedChannelTarget } from "@/shared/appTypes";
import { DISCORD_TOOL_EVIDENCE_ROLES } from "@/shared/discordTools";
import { asGuildStructureEntries, isCategoryStructureEntry } from "./guildStructure";
import type { ArgumentEnrichmentContext, GuildStructurePayload, ToolEnrichmentResult, ToolStrategy } from "./types";
import { sanitizeReason } from "./utils";

export const resolveChannelTargetsStrategy: ToolStrategy = {
    id: "resolve_channel_targets",

    extractEvidence(run: DiscordToolResult, limits): EvidenceItem[] {
        if (!run.data) {
            return [];
        }

        const item = run.data as { entries?: Array<Record<string, unknown>>; resolvedIds?: string[] };
        const entries = asGuildStructureEntries(item.entries);
        const maxItems = limits?.maxResolveChannelTargetEvidenceItems ?? 6;

        return entries.slice(0, maxItems).map((entry) => ({
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

    enrichArguments(
        modelArgs: ToolArguments,
        modelStep: { reason: string; learnedExpectation: string },
        ctx: ArgumentEnrichmentContext
    ): ToolEnrichmentResult {
        return {
            arguments: {
                targetText:
                    typeof modelArgs.targetText === "string" && (modelArgs.targetText as string).trim()
                        ? (modelArgs.targetText as string).trim()
                        : ctx.structuralChannel || ctx.activeChannelTarget?.query || ctx.question,
            },
            reason: sanitizeReason(
                modelStep.reason,
                "Resolve the referenced channel or category before answering."
            ),
            learnedExpectation: sanitizeReason(
                modelStep.learnedExpectation,
                "Return exact message-channel ids for the current guild target."
            ),
        };
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

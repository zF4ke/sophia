import type { EvidenceItem, ToolArguments } from "@/runtime/contracts";
import type { DiscordToolResult, ResolvedChannelTarget } from "@/shared/appTypes";
import { asGuildStructureEntries, buildStructureEvidenceItems } from "./guildStructure";
import type { ArgumentEnrichmentContext, GuildStructurePayload, ToolEnrichmentResult, ToolStrategy } from "./types";
import { sanitizeReason } from "./utils";

export const listGuildStructureStrategy: ToolStrategy = {
    id: "list_guild_structure",

    extractEvidence(run: DiscordToolResult): EvidenceItem[] {
        if (!run.data) {
            return [];
        }

        return buildStructureEvidenceItems(run.data as GuildStructurePayload, run.summary);
    },

    enrichArguments(
        modelArgs: ToolArguments,
        modelStep: { reason: string; learnedExpectation: string },
        ctx: ArgumentEnrichmentContext
    ): ToolEnrichmentResult {
        return {
            arguments: {
                ...(typeof modelArgs.targetText === "string" && (modelArgs.targetText as string).trim()
                    ? { targetText: (modelArgs.targetText as string).trim() }
                    : ctx.activeChannelTarget?.query
                      ? { targetText: ctx.activeChannelTarget.query }
                      : ctx.structuralChannel
                        ? { targetText: ctx.structuralChannel }
                        : { targetText: ctx.question }),
            },
            reason: sanitizeReason(
                modelStep.reason,
                "Inspect the current guild structure around the resolved or likely category/channel target."
            ),
            learnedExpectation: sanitizeReason(
                modelStep.learnedExpectation,
                "Return matched categories/channels plus visible child-channel structure."
            ),
        };
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

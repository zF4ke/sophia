import type { EvidenceItem, ToolArguments } from "@/runtime/contracts";
import type { DiscordToolResult } from "@/shared/appTypes";
import { DISCORD_TOOL_EVIDENCE_ROLES } from "@/shared/discordTools";
import type { ArgumentEnrichmentContext, ToolEnrichmentResult, ToolStrategy } from "./types";
import { sanitizeReason } from "./utils";

export const getGuildContextStrategy: ToolStrategy = {
    id: "get_guild_context",

    extractEvidence(run: DiscordToolResult): EvidenceItem[] {
        if (!run.data) {
            return [];
        }

        const item = run.data as Record<string, unknown>;
        return [
            {
                tool: "get_guild_context",
                summary: run.summary,
                content: `${String(item.name || "Guild")}: ${String(item.memberCount || 0)} members, ${String(item.channelCount || 0)} channels`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.get_guild_context,
                strength: "metadata",
                sourceOrigin: "none",
            },
        ];
    },

    enrichArguments(
        modelArgs: ToolArguments,
        modelStep: { reason: string; learnedExpectation: string },
        _ctx: ArgumentEnrichmentContext
    ): ToolEnrichmentResult {
        return {
            arguments: modelArgs,
            reason: sanitizeReason(modelStep.reason, "Use the selected capability."),
            learnedExpectation: sanitizeReason(
                modelStep.learnedExpectation,
                "Use the capability output to improve the answer."
            ),
        };
    },
};

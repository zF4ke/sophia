import type { EvidenceItem, ToolArguments } from "@/runtime/contracts";
import type { DiscordToolResult, ResolvedMemberIdentity } from "@/shared/appTypes";
import { DISCORD_TOOL_EVIDENCE_ROLES } from "@/shared/discordTools";
import type { ArgumentEnrichmentContext, ToolEnrichmentResult, ToolStrategy } from "./types";
import { sanitizeReason } from "./utils";

export const resolveMemberIdentityStrategy: ToolStrategy = {
    id: "resolve_member_identity",

    extractEvidence(run: DiscordToolResult): EvidenceItem[] {
        if (!run.data) {
            return [];
        }

        const item = run.data as Record<string, unknown>;
        const currentState =
            item.isCurrentGuildMember === false ? "historical guild memory" : "current guild";
        const resolvedId =
            item.resolvedId == null
                ? item.id == null
                    ? null
                    : String(item.id)
                : String(item.resolvedId);
        const resolveParts = [
            `${String(item.displayName || "No Display Name")} (@${String(item.username || "no username")})`,
        ];
        if (resolvedId) resolveParts.push(`id=${resolvedId}`);
        if (item.nickname) resolveParts.push(`nick=${String(item.nickname)}`);
        resolveParts.push(`from ${currentState}`);

        return [
            {
                tool: "resolve_member_identity",
                summary: run.summary,
                content: resolveParts.join("; "),
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.resolve_member_identity,
                strength: "metadata",
                sourceOrigin: "none",
                authorId: resolvedId,
                authorName: item.displayName == null ? null : String(item.displayName),
            },
        ];
    },

    enrichArguments(
        modelArgs: ToolArguments,
        modelStep: { reason: string; learnedExpectation: string },
        ctx: ArgumentEnrichmentContext
    ): ToolEnrichmentResult {
        return {
            arguments: {
                query:
                    typeof modelArgs.query === "string" && (modelArgs.query as string).trim()
                        ? (modelArgs.query as string).trim()
                        : ctx.structuralMember || ctx.activeMember?.resolvedId || ctx.actorId,
            },
            reason: sanitizeReason(
                modelStep.reason,
                "Resolve the relevant member or bot before answering."
            ),
            learnedExpectation: sanitizeReason(
                modelStep.learnedExpectation,
                "Return the best current-guild identity match or historical author fallback."
            ),
        };
    },

    extractResolvedMember(run: DiscordToolResult): ResolvedMemberIdentity | null {
        if (!run.data || typeof run.data !== "object") {
            return null;
        }

        const item = run.data as Record<string, unknown>;
        const resolvedId =
            item.resolvedId == null
                ? item.id == null
                    ? null
                    : String(item.id)
                : String(item.resolvedId);
        if (!resolvedId) {
            return null;
        }

        return {
            query: String(item.query || resolvedId),
            resolvedId,
            displayName: String(item.displayName || "No Display Name"),
            username: String(item.username || "no username"),
            globalName: item.globalName == null ? null : String(item.globalName),
            nickname: item.nickname == null ? null : String(item.nickname),
            isBot: Boolean(item.isBot),
            isCurrentGuildMember:
                item.isCurrentGuildMember == null ? true : Boolean(item.isCurrentGuildMember),
            source:
                item.source === "historical_author" ||
                item.source === "live_exact" ||
                item.source === "live_search" ||
                item.source === "live_id"
                    ? item.source
                    : "live_id",
            confidence:
                item.confidence === "high" ||
                item.confidence === "medium" ||
                item.confidence === "exact"
                    ? item.confidence
                    : "medium",
            roles: Array.isArray(item.roles) ? item.roles.map(String) : [],
        };
    },
};

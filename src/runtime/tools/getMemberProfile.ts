import type { EvidenceItem } from "@/runtime/contracts";
import type { DiscordToolResult, ResolvedMemberIdentity } from "@/shared/appTypes";
import { DISCORD_TOOL_EVIDENCE_ROLES } from "@/shared/discordTools";
import type { ToolStrategy } from "./types";

export const getMemberProfileStrategy: ToolStrategy = {
    id: "get_member_profile",

    extractEvidence(run: DiscordToolResult): EvidenceItem[] {
        if (!run.data) {
            return [];
        }

        const item = run.data as Record<string, unknown>;
        const roles = Array.isArray(item.roles) ? item.roles.join(", ") : "none";
        const parts = [
            `${String(item.displayName || "No Display Name")} (@${String(item.username || "no username")})`,
        ];
        if (item.id != null) parts.push(`id=${String(item.id)}`);
        if (item.nickname) parts.push(`nick=${String(item.nickname)}`);
        if (item.joinedAt) parts.push(`joined=${String(item.joinedAt)}`);
        if (item.accountCreatedAt) parts.push(`created=${String(item.accountCreatedAt)}`);
        parts.push(`roles=${roles || "none"}`);
        if (item.isBot) parts.push("bot=true");
        if (item.premiumSince) parts.push(`nitro_since=${String(item.premiumSince)}`);
        if (item.pending) parts.push("pending=true");

        return [
            {
                tool: "get_member_profile",
                summary: run.summary,
                content: parts.join("; "),
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.get_member_profile,
                strength: "metadata",
                sourceOrigin: "none",
                authorId: item.id == null ? null : String(item.id),
                authorName: item.displayName == null ? null : String(item.displayName),
            },
        ];
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

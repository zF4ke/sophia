import type { EvidenceItem } from "@/runtime/contracts";
import type { DiscordToolResult } from "@/shared/appTypes";
import { DISCORD_TOOL_EVIDENCE_ROLES } from "@/shared/discordTools";
import type { ToolStrategy } from "./types";

export const listMembersStrategy: ToolStrategy = {
    id: "list_members",

    extractEvidence(run: DiscordToolResult): EvidenceItem[] {
        if (!run.data || typeof run.data !== "object") {
            return [];
        }

        const payload = run.data as {
            members?: Array<Record<string, unknown>>;
            hasMore?: boolean;
            totalCount?: number;
            offset?: number;
        };
        const members = payload.members || [];

        const evidence = members.slice(0, 10).map((item) => {
            const parts = [
                `${String(item.displayName || "No Display Name")} (@${String(item.username || "no username")})`,
            ];
            if (item.id != null) parts.push(`id=${String(item.id)}`);
            if (item.nickname) parts.push(`nick=${String(item.nickname)}`);
            if (item.joinedTimestamp) {
                parts.push(`joined=${new Date(Number(item.joinedTimestamp)).toISOString()}`);
            }
            if (item.isBot) parts.push("bot=true");

            return {
                tool: "list_members" as const,
                summary: run.summary,
                content: parts.join("; "),
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_members,
                strength: "metadata" as const,
                sourceOrigin: "none" as const,
                authorId: item.id == null ? null : String(item.id),
                authorName: item.displayName == null ? null : String(item.displayName),
            };
        });

        if (payload.hasMore) {
            evidence.push({
                tool: "list_members" as const,
                summary: run.summary,
                content: `...and ${(payload.totalCount || 0) - members.length} more members (use offset=${(payload.offset || 0) + members.length} to continue).`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_members,
                strength: "metadata" as const,
                sourceOrigin: "none" as const,
                authorId: null,
                authorName: null,
            });
        }

        return evidence;
    },
};

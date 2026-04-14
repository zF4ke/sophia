import type { EvidenceItem } from "@/runtime/contracts";
import type { DiscordToolResult } from "@/shared/appTypes";
import { DISCORD_TOOL_EVIDENCE_ROLES } from "@/shared/discordTools";
import type { ToolStrategy } from "./types";

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
};

import { z } from "zod";
import { T } from "@/shared/discordTools";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import type { DiscordToolResult } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {},
    required: [],
} as const;

export const getGuildContextTool: ToolDefinition = {
    name: T.get_guild_context,

    catalog: {
        effect: "read",
        description: "Fetch live guild metadata such as name and channel counts.",
        evidenceRole: "live_evidence",
    },

    schema: {
        description:
            "Fetch live guild metadata: name, member count, channel count, creation date, icon, and guild ID.",
        parameters,
    },

    capability: {
        description: "Fetch live guild metadata.",
        inputSchema: z.object({}),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: ["guild context should exist"],
        postconditions: ["returns live guild context data"],
        async run(context) {
            const guildContext = await DiscordLiveService.getGuildContext(context.guild);
            return {
                tool: T.get_guild_context,
                summary: guildContext
                    ? `${guildContext.name}: ${guildContext.memberCount} members and ${guildContext.channelCount} channels.`
                    : "Guild context unavailable.",
                data: guildContext,
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data) return [];
            const item = run.data as Record<string, unknown>;
            return [
                {
                    tool: T.get_guild_context,
                    summary: run.summary,
                    content: `${String(item.name || "Guild")}: ${String(item.memberCount || 0)} members, ${String(item.channelCount || 0)} channels`,
                    evidenceRole: "live_evidence" as const,
                    strength: "metadata" as const,
                    sourceOrigin: "none" as const,
                },
            ];
        },
    },

    display: { icon: "🏰", labelPt: "Ler contexto do servidor" },
};

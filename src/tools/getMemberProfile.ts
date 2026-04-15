import { z } from "zod";
import { T } from "@/shared/discordTools";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import type { DiscordToolResult, ResolvedMemberIdentity } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        nameOrId: {
            type: "string",
            description: "Member name, username, nickname, or Discord snowflake ID.",
        },
    },
    required: ["nameOrId"],
} as const;

export const getMemberProfileTool: ToolDefinition = {
    name: T.get_member_profile,

    catalog: {
        effect: "read",
        description: "Fetch a member profile live from Discord.",
        evidenceRole: "live_evidence",
    },

    schema: {
        description:
            "Fetch a detailed live guild member profile. Returns roles, join date, account age, nickname, bot status, boost status (premiumSince = when they started boosting this guild, NOT Nitro), avatar URL, and Discord ID. Use the returned ID for filtering messages by author.",
        parameters,
    },

    capability: {
        description:
            "Fetch a detailed live guild member profile including roles, join date, account age, and avatar. Pass a Discord snowflake ID, a username, a display name, or a nickname — snowflake IDs resolve directly without a search.",
        inputSchema: z.object({
            nameOrId: z.string().describe("Member name, username, nickname, or Discord snowflake ID."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["returns live member identity/profile evidence"],
        async run(context, args) {
            const profile = await DiscordLiveService.getMemberProfile(
                context.guild,
                String(args.nameOrId || context.question)
            );
            return {
                tool: T.get_member_profile,
                summary: profile
                    ? [
                          `${profile.displayName} (@${profile.username})`,
                          profile.nickname ? `nick="${profile.nickname}"` : null,
                          profile.joinedAt ? `joined=${profile.joinedAt}` : null,
                          `${profile.roles.length} roles`,
                          profile.isBot ? "bot" : null,
                          profile.premiumSince ? "boosting" : null,
                      ]
                          .filter(Boolean)
                          .join(", ")
                    : "Member not found.",
                data: profile,
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data) return [];
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
            if (item.premiumSince) parts.push(`boosting_since=${String(item.premiumSince)}`);
            if (item.pending) parts.push("pending=true");

            return [
                {
                    tool: T.get_member_profile,
                    summary: run.summary,
                    content: parts.join("; "),
                    evidenceRole: "live_evidence" as const,
                    strength: "metadata" as const,
                    sourceOrigin: "none" as const,
                    authorId: item.id == null ? null : String(item.id),
                    authorName: item.displayName == null ? null : String(item.displayName),
                },
            ];
        },

        extractResolvedMember(run: DiscordToolResult): ResolvedMemberIdentity | null {
            if (!run.data || typeof run.data !== "object") return null;
            const item = run.data as Record<string, unknown>;
            const resolvedId =
                item.resolvedId == null
                    ? item.id == null ? null : String(item.id)
                    : String(item.resolvedId);
            if (!resolvedId) return null;

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
    },

    display: { icon: "👤", labelPt: "Ler perfil" },
};

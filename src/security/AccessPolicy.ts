import type { Guild } from "discord.js";
import { SettingsService } from "@/app/SettingsService";
import { SecurityService } from "./SecurityService";
import { isGuildAllowed } from "./guildAllowlist";
import type { AccessDecision } from "./accessConfig";
import type { SideEffectLevel } from "@/runtime/contracts";
import type { DiscordToolName } from "@/shared/discordTools";

export type AuthorizeAction = (effect: SideEffectLevel, tool?: DiscordToolName) => Promise<AccessDecision>;
const ranks = { none: 0, read: 0, write: 1, destructive: 2 };

/** Receives IDs from authenticated Discord events, never from model arguments. */
export class AccessPolicy {
    static forActor(userId: string, guild: Guild | null): AuthorizeAction {
        // Capture the event's immutable ID, rather than a mutable User reference.
        return (effect, tool) => this.decide(userId, guild, effect, tool);
    }

    static async decide(userId: string, guild: Guild | null, effect: SideEffectLevel, tool?: DiscordToolName): Promise<AccessDecision> {
        if (!userId || !isGuildAllowed(guild?.id)) return "deny";
        await SecurityService.initialize();
        const settings = SettingsService.load();
        const access = settings.access;
        const active = (grant: { expiresAt?: string }) => !grant.expiresAt || Date.parse(grant.expiresAt) > Date.now();
        const grants = access.users.filter(grant => active(grant) && grant.userId === userId &&
            (!grant.guildId || grant.guildId === guild?.id));
        const roleGrants = access.roles.filter(grant => active(grant) && grant.guildId === guild?.id);
        const candidates = (access.rules ?? []).filter(rule => (!rule.guildId || rule.guildId === guild?.id) &&
            (!rule.tool || rule.tool === tool) && (!rule.tier || rule.tier === (effect === "none" ? "read" : effect)));
        let memberRoles: { has(id: string): boolean } | undefined;
        if (guild && (roleGrants.length > 0 || candidates.some(rule => rule.subject === "role"))) {
            // Force refresh so a revoked role does not survive a queued operation.
            const member = await guild.members.fetch({ user: userId, force: true }).catch(() => null);
            // Missing membership cannot safely exclude an explicit role deny.
            if (!member && candidates.some(rule => rule.subject === "role" && rule.decision === "deny")) return "deny";
            memberRoles = member?.roles.cache;
            for (const grant of roleGrants) {
                if (member?.roles.cache.has(grant.roleId)) grants.push({ userId, ...grant });
            }
        }
        const permitted = grants.filter(grant => active(grant) && ranks[grant.level] >= ranks[effect]);
        const rules = candidates.filter(rule => rule.subject === "user" ? rule.subjectId === userId : memberRoles?.has(rule.subjectId));
        if (rules.some(rule => rule.decision === "deny")) return "deny";
        const operator = SecurityService.isAdmin(userId);
        if (!operator && !(effect === "none" && SecurityService.isModerator(userId)) && !permitted.length) return "deny";
        // Rules can change approval within a grant, never grant a higher action tier.
        if (effect === "none") return "allow";
        if (rules.some(rule => rule.decision === "ask")) return "ask";
        if (rules.some(rule => rule.decision === "allow")) return "allow";
        return permitted.some(grant => grant.mode === "auto") ? "allow" : "ask";
    }
}

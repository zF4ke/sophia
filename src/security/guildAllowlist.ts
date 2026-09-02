import { SettingsService } from "@/app/SettingsService";

/**
 * Guild allowlist gate. When `guildAllowlist` is non-empty in settings,
 * only messages, interactions, and crawls from listed guilds are processed.
 * Empty list = all guilds allowed (default, backward compatible).
 */
export function isGuildAllowed(guildId: string | null | undefined): boolean {
    if (!guildId) return true;
    try {
        const allowlist = SettingsService.load().guildAllowlist;
        if (!Array.isArray(allowlist) || allowlist.length === 0) return true;
        return allowlist.includes(guildId);
    } catch {
        return true;
    }
}

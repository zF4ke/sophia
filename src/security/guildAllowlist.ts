import { SettingsService } from "@/app/SettingsService";

/**
 * Admission for bot work. An empty list enables nothing; errors fail closed.
 */
export function isGuildAllowed(guildId: string | null | undefined): boolean {
    try {
        const settings = SettingsService.load();
        if (!guildId) return settings.access.directMessages;
        const allowlist = settings.guildAllowlist;
        if (!Array.isArray(allowlist)) return false;
        return allowlist.includes(guildId);
    } catch {
        return false;
    }
}

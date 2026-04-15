import { SettingsService } from "@/app/SettingsService";

/**
 * Channels in this set are immune to all destructive tool calls.
 * Destructive actions can still go through the approval flow for visibility,
 * but execution is auto-blocked afterwards if the target channel is protected.
 */
export class ProtectedChannelsService {
    public static getAll(): Set<string> {
        return new Set(SettingsService.load().protectedChannelIds ?? []);
    }

    public static isProtected(channelId: string): boolean {
        return this.getAll().has(channelId);
    }

    public static add(channelIds: string[]): string[] {
        const current = SettingsService.load();
        const existing = new Set(current.protectedChannelIds ?? []);
        for (const id of channelIds) existing.add(id);
        const updated = [...existing];
        SettingsService.update({ protectedChannelIds: updated });
        return updated;
    }

    public static remove(channelIds: string[]): string[] {
        const current = SettingsService.load();
        const existing = new Set(current.protectedChannelIds ?? []);
        for (const id of channelIds) existing.delete(id);
        const updated = [...existing];
        SettingsService.update({ protectedChannelIds: updated });
        return updated;
    }

    public static clear(): void {
        SettingsService.update({ protectedChannelIds: [] });
    }
}

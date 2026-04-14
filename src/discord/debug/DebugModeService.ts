import { SettingsService } from "@/app/SettingsService";

export class DebugModeService {
    private static enabled: boolean | null = null;

    public static isEnabled(): boolean {
        if (this.enabled === null) {
            this.enabled = SettingsService.load().debug;
        }

        return this.enabled;
    }

    public static setEnabled(enabled: boolean): void {
        this.enabled = enabled;
        SettingsService.update({ debug: enabled });
    }

    public static resetForTests(): void {
        this.enabled = null;
    }
}

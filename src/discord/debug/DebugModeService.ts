import { DebugStateStore } from "@/discord/debug/DebugStateStore";

export class DebugModeService {
    private static enabled: boolean | null = null;

    public static isEnabled(): boolean {
        if (this.enabled === null) {
            this.enabled = DebugStateStore.load().enabled;
        }

        return this.enabled;
    }

    public static setEnabled(enabled: boolean): void {
        this.enabled = enabled;
        DebugStateStore.save({ enabled });
    }

    public static resetForTests(): void {
        this.enabled = null;
    }
}

export class StatusFormatter {
    public static format(emoji: string, message: string, useBackticks = true): string {
        return useBackticks ? `\`${emoji} ${message}\`` : `${emoji} ${message}`;
    }
}

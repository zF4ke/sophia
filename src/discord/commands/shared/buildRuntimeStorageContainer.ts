import { ContainerBuilder, TextDisplayBuilder } from "discord.js";

interface RuntimeStorageStatus {
    operationalDbPath: string;
    checkpointDbPath: string;
    operationalDbSizeBytes: number;
    checkpointDbSizeBytes: number;
    operationalSchemaVersion: string;
    runtimeDir: string;
    logsDir: string;
}

function formatBytes(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return "0 B";
    }

    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    const digits = value >= 100 || unitIndex === 0 ? 0 : value >= 10 ? 1 : 2;
    return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

export function buildRuntimeStorageContainer(status: RuntimeStorageStatus): ContainerBuilder {
    const totalSizeBytes =
        status.operationalDbSizeBytes + status.checkpointDbSizeBytes;
    return new ContainerBuilder()
        .setAccentColor(0x57f287)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## Runtime storage"),
            new TextDisplayBuilder().setContent(
                [
                    `**Operational DB:** ${status.operationalDbPath} (${formatBytes(status.operationalDbSizeBytes)})`,
                    `**Checkpoint DB:** ${status.checkpointDbPath} (${formatBytes(status.checkpointDbSizeBytes)})`,
                    `**Total DB size:** ${formatBytes(totalSizeBytes)}`,
                    `**Operational schema:** ${status.operationalSchemaVersion}`,
                    `**Runtime dir:** ${status.runtimeDir}`,
                    `**Logs dir:** ${status.logsDir}`,
                ].join("\n")
            )
        );
}

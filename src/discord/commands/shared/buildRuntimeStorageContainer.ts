import { ContainerBuilder, TextDisplayBuilder } from "discord.js";

interface RuntimeStorageStatus {
    operationalDbPath: string;
    checkpointDbPath: string;
    operationalSchemaVersion: string;
    checkpointSchemaVersion: string;
    checkpointVersionPath: string;
    runtimeDir: string;
    logsDir: string;
}

export function buildRuntimeStorageContainer(status: RuntimeStorageStatus): ContainerBuilder {
    return new ContainerBuilder()
        .setAccentColor(0x57f287)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## Runtime storage"),
            new TextDisplayBuilder().setContent(
                [
                    `**Operational DB:** ${status.operationalDbPath}`,
                    `**Checkpoint DB:** ${status.checkpointDbPath}`,
                    `**Operational schema:** ${status.operationalSchemaVersion}`,
                    `**Checkpoint schema:** ${status.checkpointSchemaVersion}`,
                    `**Checkpoint version file:** ${status.checkpointVersionPath}`,
                    `**Runtime dir:** ${status.runtimeDir}`,
                    `**Logs dir:** ${status.logsDir}`,
                ].join("\n")
            )
        );
}

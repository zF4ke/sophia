import { ContainerBuilder, TextDisplayBuilder } from "discord.js";

export function buildGuildCompletenessContainer(summary: {
    liveReadableChannelCount: number;
    liveReadableCategoryCount: number;
    cachedOnlyCount: number;
    indexedChannelCount: number;
    liveHydrated: boolean;
}): ContainerBuilder {
    return new ContainerBuilder()
        .setAccentColor(0xf1c40f)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## Current guild completeness"),
            new TextDisplayBuilder().setContent(
                [
                    `**Live guild hydrated:** ${summary.liveHydrated ? "yes" : "no"}`,
                    `**Readable live channels:** ${summary.liveReadableChannelCount}`,
                    `**Readable live categories:** ${summary.liveReadableCategoryCount}`,
                    `**Cached-only remembered entries:** ${summary.cachedOnlyCount}`,
                    `**Indexed channels:** ${summary.indexedChannelCount}`,
                ].join("\n")
            )
        );
}

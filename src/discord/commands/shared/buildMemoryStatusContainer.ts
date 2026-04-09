import {
    ContainerBuilder,
    SeparatorBuilder,
    SeparatorSpacingSize,
    TextDisplayBuilder,
} from "discord.js";
import type { ChannelIndexState } from "@/memory/types";

interface MemoryStats {
    messages: number;
    chunks: number;
    channels: number;
}

const numberFormatter = new Intl.NumberFormat("pt-PT");

export function buildMemoryStatusContainer(
    stats: MemoryStats,
    states: ChannelIndexState[],
    maxChannels = 6
): ContainerBuilder {
    const sortedStates = [...states].sort(
        (left, right) =>
            (right.lastIndexedTimestamp ?? 0) - (left.lastIndexedTimestamp ?? 0)
    );
    const visibleStates = sortedStates.slice(0, maxChannels);
    const hiddenCount = Math.max(sortedStates.length - visibleStates.length, 0);
    const channelLines = visibleStates.length
        ? visibleStates.map((state) =>
              state.lastIndexedTimestamp
                  ? `<#${state.channelId}> atualizado <t:${Math.floor(
                        state.lastIndexedTimestamp / 1000
                    )}:R>`
                  : `<#${state.channelId}> sem histórico indexado`
          )
        : ["Nenhum canal foi indexado ainda."];

    const container = new ContainerBuilder()
        .setAccentColor(0x9aa7ff)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## Estado da memória local"),
            new TextDisplayBuilder().setContent(
                "Blocos pesquisáveis são partes das mensagens usadas na busca."
            ),
            new TextDisplayBuilder().setContent(
                [
                    `**Mensagens salvas:** ${numberFormatter.format(stats.messages)}`,
                    `**Blocos pesquisáveis:** ${numberFormatter.format(stats.chunks)}`,
                    `**Canais indexados:** ${numberFormatter.format(stats.channels)}`,
                ].join("\n")
            )
        )
        .addSeparatorComponents(
            new SeparatorBuilder()
                .setDivider(true)
                .setSpacing(SeparatorSpacingSize.Small)
        )
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("### Canais"),
            new TextDisplayBuilder().setContent(channelLines.join("\n"))
        );

    if (hiddenCount > 0) {
        container.addTextDisplayComponents(
            new TextDisplayBuilder().setContent(
                `Mais ${hiddenCount} canal${hiddenCount === 1 ? "" : "is"} indexado${hiddenCount === 1 ? "" : "s"} não mostrado${hiddenCount === 1 ? "" : "s"}.`
            )
        );
    }

    return container;
}

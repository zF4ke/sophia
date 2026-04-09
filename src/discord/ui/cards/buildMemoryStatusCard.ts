import {
    ContainerBuilder,
    SectionBuilder,
    SeparatorBuilder,
    SeparatorSpacingSize,
    TextDisplayBuilder,
    ThumbnailBuilder,
} from "discord.js";
import type { Client } from "discord.js";
import type { ChannelIndexState } from "@/memory/types";

interface MemoryStats {
    messages: number;
    chunks: number;
    channels: number;
}

const numberFormatter = new Intl.NumberFormat("pt-PT");

function formatMetric(label: string, value: number): string {
    return `**${label}**\n\`${numberFormatter.format(value)}\``;
}

function formatChannelState(state: ChannelIndexState): string {
    if (!state.lastIndexedTimestamp) {
        return `- <#${state.channelId}> · ainda sem histórico indexado`;
    }

    return `- <#${state.channelId}> · atualizado <t:${Math.floor(state.lastIndexedTimestamp / 1000)}:R>`;
}

export function buildMemoryStatusCard(
    client: Client,
    stats: MemoryStats,
    states: ChannelIndexState[],
    hiddenCount: number
): ContainerBuilder {
    const channelSummary = states.length
        ? states.map(formatChannelState).join("\n")
        : "Nenhum canal foi indexado ainda.";

    const footer =
        hiddenCount > 0
            ? `- A mostrar os ${states.length} canais mais recentes. ${hiddenCount} canais ficaram fora deste resumo.`
            : "- Este painel mostra apenas o estado útil da memória; os cursores internos ficam ocultos.";

    return new ContainerBuilder()
        .setAccentColor(0x9aa7ff)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## Estado da memória local"),
            new TextDisplayBuilder().setContent(
                "Resumo do índice local usado pela Sophia. `Blocos pesquisáveis` são fragmentos de mensagens preparados para busca."
            )
        )
        .addSeparatorComponents(
            new SeparatorBuilder()
                .setDivider(true)
                .setSpacing(SeparatorSpacingSize.Small)
        )
        .addSectionComponents(
            new SectionBuilder()
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(
                        [
                            formatMetric("Mensagens salvas", stats.messages),
                            formatMetric("Blocos pesquisáveis", stats.chunks),
                            formatMetric("Canais indexados", stats.channels),
                        ].join("\n\n")
                    )
                )
                .setThumbnailAccessory(
                    new ThumbnailBuilder()
                        .setURL(
                            client.user?.displayAvatarURL({
                                extension: "png",
                                size: 256,
                            }) ?? "https://cdn.discordapp.com/embed/avatars/0.png"
                        )
                        .setDescription("Sophia avatar")
                )
        )
        .addSeparatorComponents(
            new SeparatorBuilder()
                .setDivider(true)
                .setSpacing(SeparatorSpacingSize.Small)
        )
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("### Canais acompanhados"),
            new TextDisplayBuilder().setContent(channelSummary)
        )
        .addSeparatorComponents(
            new SeparatorBuilder()
                .setDivider(false)
                .setSpacing(SeparatorSpacingSize.Small)
        )
        .addTextDisplayComponents(new TextDisplayBuilder().setContent(footer));
}

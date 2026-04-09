import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ChatInputCommandInteraction,
    ComponentType,
    EmbedBuilder,
    MessageFlags,
    StringSelectMenuBuilder,
} from "discord.js";
import type { RetrievedChunk } from "@/shared/appTypes";
import { SecurityService } from "@/security/SecurityService";

type ViewState = {
    page: number;
    selectedIndex: number;
};

const PAGE_SIZE = 5;

export class GroundedResultUIService {
    public static async showSearchResults(
        interaction: ChatInputCommandInteraction,
        title: string,
        query: string,
        results: RetrievedChunk[]
    ): Promise<void> {
        const state: ViewState = {
            page: 0,
            selectedIndex: 0,
        };

        const createPayload = () => {
            const start = state.page * PAGE_SIZE;
            const currentPageItems = results.slice(start, start + PAGE_SIZE);
            const selected = currentPageItems[state.selectedIndex] || currentPageItems[0];
            const embed = new EmbedBuilder()
                .setColor(0x5865f2)
                .setTitle(title)
                .setDescription(
                    `Consulta: **${query}**\nResultados: **${results.length}**\nCanal: **${selected?.channelName || "n/a"}**`
                )
                .setFooter({
                    text: `Página ${state.page + 1} de ${Math.max(1, Math.ceil(results.length / PAGE_SIZE))}`,
                });

            if (selected) {
                embed.addFields(
                    {
                        name: "Mensagem",
                        value: selected.content.slice(0, 1024),
                    },
                    {
                        name: "Autor",
                        value: selected.authorName,
                        inline: true,
                    },
                    {
                        name: "Pontuação",
                        value: selected.totalScore.toFixed(3),
                        inline: true,
                    },
                    {
                        name: "Link",
                        value: `[Abrir mensagem](${selected.jumpLink})`,
                    }
                );
            }

            const navRow = new ActionRowBuilder<ButtonBuilder>().addComponents(
                new ButtonBuilder()
                    .setCustomId("results_prev")
                    .setLabel("Anterior")
                    .setStyle(ButtonStyle.Secondary)
                    .setDisabled(state.page === 0),
                new ButtonBuilder()
                    .setCustomId("results_next")
                    .setLabel("Próxima")
                    .setStyle(ButtonStyle.Secondary)
                    .setDisabled(start + PAGE_SIZE >= results.length)
            );

            const selectRow =
                currentPageItems.length > 0
                    ? new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                          new StringSelectMenuBuilder()
                              .setCustomId("results_select")
                              .setPlaceholder("Selecionar evidência")
                              .addOptions(
                                  currentPageItems.map((item, index) => ({
                                      label: `${item.channelName} · ${item.authorName}`.slice(0, 100),
                                      value: String(index),
                                      description: item.content.slice(0, 100),
                                      default: index === state.selectedIndex,
                                  }))
                              )
                      )
                    : null;

            return {
                embeds: [embed],
                components: selectRow ? [selectRow, navRow] : [navRow],
            };
        };

        const reply = await interaction.editReply(createPayload());
        const collector = reply.createMessageComponentCollector({
            componentType: ComponentType.Button,
            time: 300000,
        });
        const selectCollector = reply.createMessageComponentCollector({
            componentType: ComponentType.StringSelect,
            time: 300000,
        });

        const isAllowed = (userId: string) =>
            userId === interaction.user.id || SecurityService.isAdmin(userId);

        collector.on("collect", async (component) => {
            if (!isAllowed(component.user.id)) {
                await component.reply({
                    content: "Apenas o autor do comando e administradores podem usar esses controles.",
                    flags: MessageFlags.Ephemeral,
                });
                return;
            }

            if (component.customId === "results_prev" && state.page > 0) {
                state.page -= 1;
                state.selectedIndex = 0;
            }
            if (
                component.customId === "results_next" &&
                (state.page + 1) * PAGE_SIZE < results.length
            ) {
                state.page += 1;
                state.selectedIndex = 0;
            }

            await component.update(createPayload());
        });

        selectCollector.on("collect", async (component) => {
            if (!isAllowed(component.user.id)) {
                await component.reply({
                    content: "Apenas o autor do comando e administradores podem usar esses controles.",
                    flags: MessageFlags.Ephemeral,
                });
                return;
            }

            state.selectedIndex = Number(component.values[0] || 0);
            await component.update(createPayload());
        });
    }
}

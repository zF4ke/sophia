import {
    ActionRowBuilder,
    ButtonBuilder,
    ComponentType,
    EmbedBuilder,
    Message,
    MessageFlags,
    type ChatInputCommandInteraction,
} from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import { EMOJIS } from "@/discord/constants";
import type { NavigationState } from "@/discord/ui/types";

export class NavigationCollector {
    public static readonly MAX_ITEMS_PER_PAGE = 5;
    public static readonly COLLECTOR_TIMEOUT = 300000;

    public static async collect<T>(
        message: Message,
        interaction: ChatInputCommandInteraction,
        items: T[],
        state: NavigationState,
        createEmbed: (state: NavigationState) => EmbedBuilder,
        createRow: (state: NavigationState) => ActionRowBuilder<ButtonBuilder>
    ): Promise<void> {
        const collector = message.createMessageComponentCollector({
            componentType: ComponentType.Button,
            time: this.COLLECTOR_TIMEOUT,
        });

        collector.on("collect", async (component: any) => {
            if (
                component.user.id !== interaction.user.id &&
                !SecurityService.isAdmin(component.user.id)
            ) {
                await component.reply({
                    content: `${EMOJIS.error} Apenas o autor do comando e administradores podem interagir com esses botões.`,
                    flags: MessageFlags.Ephemeral,
                });
                return;
            }

            let updated = false;
            switch (component.customId) {
                case "prev_conv":
                    if (state.currentConvIndex > 0) {
                        state.currentConvIndex -= 1;
                        state.currentMsgIndex = 0;
                        updated = true;
                    }
                    break;
                case "next_conv":
                    if (state.currentConvIndex < items.length - 1) {
                        state.currentConvIndex += 1;
                        state.currentMsgIndex = 0;
                        updated = true;
                    }
                    break;
                case "prev_page":
                    if (state.currentMsgIndex > 0) {
                        state.currentMsgIndex = Math.max(
                            0,
                            state.currentMsgIndex - this.MAX_ITEMS_PER_PAGE
                        );
                        updated = true;
                    }
                    break;
                case "next_page": {
                    const maxIndex = (items[state.currentConvIndex] as any).messages?.length ?? items.length;
                    if (state.currentMsgIndex + this.MAX_ITEMS_PER_PAGE < maxIndex) {
                        state.currentMsgIndex += this.MAX_ITEMS_PER_PAGE;
                        updated = true;
                    }
                    break;
                }
            }

            if (updated) {
                await component.update({
                    embeds: [createEmbed(state)],
                    components: [createRow(state)],
                });
            }
        });

        collector.on("end", async (_, reason) => {
            if (reason === "messageDeleted") {
                return;
            }

            try {
                await interaction.editReply({
                    embeds: [
                        createEmbed(state).setFooter({
                            text: "Esta sessão expirou. Execute o comando novamente para uma nova sessão.",
                        }),
                    ],
                    components: [],
                });
            } catch (error: any) {
                if (error.code !== 10008) {
                    console.error("Error updating expired interaction:", error);
                }
            }
        });
    }
}

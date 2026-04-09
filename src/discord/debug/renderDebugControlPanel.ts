import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    TextDisplayBuilder,
} from "discord.js";

export const DEBUG_ENABLE_ID = "debug:enable";
export const DEBUG_DISABLE_ID = "debug:disable";

export function renderDebugControlPanel(
    enabled: boolean,
    notice?: string
): {
    components: [ContainerBuilder, ActionRowBuilder<ButtonBuilder>];
} {
    const container = new ContainerBuilder()
        .setAccentColor(enabled ? 0x57f287 : 0x9aa7ff)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## Painel de debug"),
            new TextDisplayBuilder().setContent(
                [
                    `**Estado atual:** ${enabled ? "Ativado" : "Desativado"}`,
                    "**Afeta:** /ask, /talk, /context, menções à Sophia e respostas para a Sophia.",
                    "**Saída:** uma mensagem pública de debug por resposta, atualizada ao longo do fluxo.",
                ].join("\n")
            )
        );

    if (notice) {
        container.addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`**Atualização:** ${notice}`)
        );
    }

    const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(DEBUG_ENABLE_ID)
            .setLabel("Ativar")
            .setStyle(ButtonStyle.Success)
            .setDisabled(enabled),
        new ButtonBuilder()
            .setCustomId(DEBUG_DISABLE_ID)
            .setLabel("Desativar")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(!enabled)
    );

    return {
        components: [container, row],
    };
}

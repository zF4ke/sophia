import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    TextDisplayBuilder,
} from "discord.js";

export const DEBUG_ENABLE_ID = "debug:enable";
export const DEBUG_DISABLE_ID = "debug:disable";

type DebugControlPanelOptions = {
    enableId?: string;
    disableId?: string;
    title?: string;
};

export function renderDebugControlPanel(
    enabled: boolean,
    notice?: string,
    options: DebugControlPanelOptions = {}
): {
    components: [ContainerBuilder, ActionRowBuilder<ButtonBuilder>];
} {
    const enableId = options.enableId ?? DEBUG_ENABLE_ID;
    const disableId = options.disableId ?? DEBUG_DISABLE_ID;
    const title = options.title ?? "## Painel de Debug";
    const container = new ContainerBuilder()
        .setAccentColor(enabled ? 0x57f287 : 0x9aa7ff)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(title),
            new TextDisplayBuilder().setContent(
                [
                    `**Estado atual:** ${enabled ? "Ativo" : "Inativo"}`,
                    "**Afeta:** /talk, menções e respostas à Sophia.",
                    "**Saída:** rasto público por resposta, atualizado durante o fluxo.",
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
            .setCustomId(enableId)
            .setLabel("Ativar")
            .setStyle(ButtonStyle.Success)
            .setDisabled(enabled),
        new ButtonBuilder()
            .setCustomId(disableId)
            .setLabel("Desativar")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(!enabled)
    );

    return {
        components: [container, row],
    };
}

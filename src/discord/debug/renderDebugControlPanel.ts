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
            new TextDisplayBuilder().setContent("## Debug Panel"),
            new TextDisplayBuilder().setContent(
                [
                    `**Current state:** ${enabled ? "Enabled" : "Disabled"}`,
                    "**Affects:** /talk, mentions, and replies to Sophia.",
                    "**Output:** a public debug trace per response, updated throughout the flow.",
                ].join("\n")
            )
        );

    if (notice) {
        container.addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`**Update:** ${notice}`)
        );
    }

    const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(DEBUG_ENABLE_ID)
            .setLabel("Enable")
            .setStyle(ButtonStyle.Success)
            .setDisabled(enabled),
        new ButtonBuilder()
            .setCustomId(DEBUG_DISABLE_ID)
            .setLabel("Disable")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(!enabled)
    );

    return {
        components: [container, row],
    };
}

import {
    ChatInputCommandInteraction,
    ContainerBuilder,
    MessageFlags,
    SectionBuilder,
    SeparatorBuilder,
    SeparatorSpacingSize,
    SlashCommandBuilder,
    TextDisplayBuilder,
    ThumbnailBuilder,
} from "discord.js";

function getLatencyTone(latencyMs: number, gatewayMs: number): {
    accentColor: number;
    label: string;
} {
    const worst = Math.max(latencyMs, gatewayMs);

    if (worst <= 150) {
        return {
            accentColor: 0x57f287,
            label: "Healthy",
        };
    }

    if (worst <= 350) {
        return {
            accentColor: 0xfee75c,
            label: "Stable",
        };
    }

    return {
        accentColor: 0xed4245,
        label: "Slow",
    };
}

export = {
    data: new SlashCommandBuilder()
        .setName("ping")
        .setDescription("Verifica se a Sophia está online")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addBooleanOption((option) =>
            option
                .setName("ephemeral")
                .setDescription("Mostrar apenas para você")
                .setRequired(false)
        ),
    async execute(interaction: ChatInputCommandInteraction) {
        const roundTripMs = Date.now() - interaction.createdTimestamp;
        const gatewayMs = Math.round(interaction.client.ws.ping);
        const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
        const tone = getLatencyTone(roundTripMs, gatewayMs);

        const container = new ContainerBuilder()
            .setAccentColor(tone.accentColor)
            .addTextDisplayComponents(
                new TextDisplayBuilder().setContent("## Pong"),
                new TextDisplayBuilder().setContent(
                    `Sophia is online. Current status: **${tone.label}**.`
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
                                `**Round trip**\n\`${roundTripMs}ms\``,
                                `**Gateway**\n\`${gatewayMs}ms\``,
                                `**Shard**\n\`${interaction.guild?.shardId ?? 0}\``,
                            ].join("\n\n")
                        )
                    )
                    .setThumbnailAccessory(
                        new ThumbnailBuilder()
                            .setURL(
                                interaction.client.user.displayAvatarURL({
                                    extension: "png",
                                    size: 256,
                                })
                            )
                            .setDescription("Sophia avatar")
                    )
            )
            .addSeparatorComponents(
                new SeparatorBuilder()
                    .setDivider(false)
                    .setSpacing(SeparatorSpacingSize.Small)
            )
            .addTextDisplayComponents(
                new TextDisplayBuilder().setContent(
                    `- Checked <t:${Math.floor(Date.now() / 1000)}:R>`
                )
            );

        await interaction.reply({
            components: [container],
            flags:
                MessageFlags.IsComponentsV2 |
                (ephemeral ? MessageFlags.Ephemeral : 0),
        });
    },
};

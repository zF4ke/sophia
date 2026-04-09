import {
    ChatInputCommandInteraction,
    ContainerBuilder,
    MessageFlags,
    SeparatorBuilder,
    SeparatorSpacingSize,
    SlashCommandBuilder,
    TextDisplayBuilder,
} from "discord.js";

function getLatencyTone(latencyMs: number, gatewayMs: number): {
    accentColor: number;
    label: string;
} {
    const worst = Math.max(latencyMs, gatewayMs);

    if (worst <= 150) {
        return {
            accentColor: 0x57f287,
            label: "Estável",
        };
    }

    if (worst <= 350) {
        return {
            accentColor: 0xfee75c,
            label: "Aceitável",
        };
    }

    return {
        accentColor: 0xed4245,
        label: "Lento",
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
        const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
        await interaction.deferReply({
            flags: ephemeral ? MessageFlags.Ephemeral : undefined,
        });

        const roundTripMs = Math.max(0, Date.now() - interaction.createdTimestamp);
        const gatewayMs = Math.max(0, Math.round(interaction.client.ws.ping));
        const tone = getLatencyTone(roundTripMs, gatewayMs);

        const container = new ContainerBuilder()
            .setAccentColor(tone.accentColor)
            .addTextDisplayComponents(
                new TextDisplayBuilder().setContent("## Pong"),
                new TextDisplayBuilder().setContent(
                    `Sophia está online. Estado atual: **${tone.label}**.`
                ),
                new TextDisplayBuilder().setContent(
                    [
                        `**Ida e volta:** ${roundTripMs}ms`,
                        `**Gateway:** ${gatewayMs}ms`,
                        `**Shard:** ${interaction.guild?.shardId ?? 0}`,
                    ].join("\n")
                )
            )
            .addSeparatorComponents(
                new SeparatorBuilder()
                    .setDivider(true)
                    .setSpacing(SeparatorSpacingSize.Small)
            )
            .addTextDisplayComponents(
                new TextDisplayBuilder().setContent(
                    `Verificado <t:${Math.floor(Date.now() / 1000)}:R>.`
                )
            );

        await interaction.editReply({
            components: [container],
            flags: MessageFlags.IsComponentsV2,
        });
    },
};

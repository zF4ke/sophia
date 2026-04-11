import {
    ChannelType,
    ChatInputCommandInteraction,
    ContainerBuilder,
    MessageFlags,
    SeparatorBuilder,
    SeparatorSpacingSize,
    SlashCommandBuilder,
    TextDisplayBuilder,
} from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

function formatMessageBody(content: string): string {
    const trimmed = content.trim();
    if (!trimmed) {
        return "_Sem conteúdo de texto._";
    }

    return trimmed.length > 1500 ? `${trimmed.slice(0, 1497)}...` : trimmed;
}

export = {
    data: new SlashCommandBuilder()
        .setName("nth")
        .setDescription("Mostra a N-ésima mensagem histórica de um canal indexado")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addChannelOption((option) =>
            option
                .setName("channel")
                .setDescription("Canal indexado")
                .addChannelTypes(
                    ChannelType.GuildText,
                    ChannelType.PublicThread,
                    ChannelType.PrivateThread
                )
                .setRequired(true)
        )
        .addIntegerOption((option) =>
            option
                .setName("number")
                .setDescription("Posição histórica da mensagem")
                .setRequired(true)
                .setMinValue(1)
        )
        .addBooleanOption((option) =>
            option
                .setName("ephemeral")
                .setDescription("Somente você pode ver o resultado")
                .setRequired(false)
        ),
    async execute(interaction: ChatInputCommandInteraction) {
        const channel = interaction.options.getChannel("channel", true);
        const number = interaction.options.getInteger("number", true);
        const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;

        await interaction.deferReply({
            flags: ephemeral ? MessageFlags.Ephemeral : undefined,
        });

        const stored = await DiscordMemoryService.getNthHistoricalMessageAsync(channel.id, number);
        if (!stored) {
            await interaction.editReply({
                components: [
                    new ContainerBuilder()
                        .setAccentColor(0xfee75c)
                        .addTextDisplayComponents(
                            new TextDisplayBuilder().setContent("## Mensagem não encontrada"),
                            new TextDisplayBuilder().setContent(
                                "Essa posição ainda não existe na memória local. Faça backfill desse canal antes de pedir mensagens mais antigas."
                            )
                        ),
                ],
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        await interaction.editReply({
            components: [
                new ContainerBuilder()
                    .setAccentColor(0x9aa7ff)
                    .addTextDisplayComponents(
                        new TextDisplayBuilder().setContent(
                            `## Mensagem #${number} em <#${channel.id}>`
                        ),
                        new TextDisplayBuilder().setContent(
                            [
                                `**Autor:** ${stored.authorName}`,
                                `**Data:** <t:${Math.floor(stored.createdTimestamp / 1000)}:f>`,
                                `**Link:** [Abrir mensagem](${stored.jumpLink})`,
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
                            formatMessageBody(stored.content)
                        )
                    ),
            ],
            flags: MessageFlags.IsComponentsV2,
        });
    },
};

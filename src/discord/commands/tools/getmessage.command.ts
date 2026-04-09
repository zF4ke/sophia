import {
    ChannelType,
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { UIService } from "@/discord/ui/UIService";
import { EMOJIS } from "@/discord/constants";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("getmessage")
        .setDescription("Obtém a N-ésima mensagem histórica de um canal já indexado")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addChannelOption((option) =>
            option
                .setName("channel")
                .setDescription("Canal indexado")
                .addChannelTypes(ChannelType.GuildText, ChannelType.PublicThread, ChannelType.PrivateThread)
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
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        const channel = interaction.options.getChannel("channel", true);
        const number = interaction.options.getInteger("number", true);
        const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;

        await interaction.deferReply({
            flags: ephemeral ? MessageFlags.Ephemeral : undefined,
        });

        const stored = DiscordMemoryService.getNthHistoricalMessage(channel.id, number);
        if (!stored) {
            await interaction.editReply(
                UIService.formatStatusMessage(
                    EMOJIS.warning,
                    "Essa mensagem não existe na memória local. Rode o backfill do canal antes.",
                    false
                )
            );
            return;
        }

        await interaction.editReply({
            content:
                `Mensagem #${number} em <#${channel.id}>:\n\n` +
                `Autor: **${stored.authorName}**\n` +
                `Data: <t:${Math.floor(stored.createdTimestamp / 1000)}:f>\n` +
                `Conteúdo:\n${stored.content}\n\n` +
                `[Abrir mensagem](${stored.jumpLink})`,
        });
    },
};

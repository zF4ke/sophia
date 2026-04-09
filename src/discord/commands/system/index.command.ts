import {
    CategoryChannel,
    ChannelType,
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
    TextChannel,
    ThreadChannel,
} from "discord.js";
import { DiscordBackfillService } from "@/discord/live/DiscordBackfillService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { SecurityService } from "@/security/SecurityService";
import { UIService } from "@/discord/ui/UIService";
import { EMOJIS } from "@/discord/constants";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("index")
        .setDescription("Gerenciar a memória local do Discord")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addSubcommand((subcommand) =>
            subcommand
                .setName("status")
                .setDescription("Ver o estado atual do índice")
        )
        .addSubcommand((subcommand) =>
            subcommand
                .setName("clear")
                .setDescription("Apagar toda a memória local indexada")
        )
        .addSubcommand((subcommand) =>
            subcommand
                .setName("repair")
                .setDescription("Reparar índices FTS e embeddings persistidos")
        )
        .addSubcommand((subcommand) =>
            subcommand
                .setName("backfill_channel")
                .setDescription("Indexar histórico de um canal")
                .addChannelOption((option) =>
                    option
                        .setName("channel")
                        .setDescription("Canal a indexar")
                        .addChannelTypes(ChannelType.GuildText, ChannelType.PublicThread, ChannelType.PrivateThread)
                        .setRequired(true)
                )
                .addIntegerOption((option) =>
                    option
                        .setName("limit")
                        .setDescription("Quantidade máxima de mensagens")
                        .setRequired(false)
                        .setMinValue(1)
                        .setMaxValue(5000)
                )
        )
        .addSubcommand((subcommand) =>
            subcommand
                .setName("backfill_category")
                .setDescription("Indexar histórico dos canais de uma categoria")
                .addChannelOption((option) =>
                    option
                        .setName("category")
                        .setDescription("Categoria a indexar")
                        .addChannelTypes(ChannelType.GuildCategory)
                        .setRequired(true)
                )
                .addIntegerOption((option) =>
                    option
                        .setName("limit_per_channel")
                        .setDescription("Quantidade máxima por canal")
                        .setRequired(false)
                        .setMinValue(1)
                        .setMaxValue(5000)
                )
        ),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                flags: MessageFlags.Ephemeral,
            });
            return;
        }

        await interaction.deferReply({ flags: MessageFlags.Ephemeral });
        const subcommand = interaction.options.getSubcommand();

        if (subcommand === "status") {
            const stats = DiscordMemoryService.getStats();
            const states = DiscordMemoryService.getIndexState().slice(0, 15);
            const lines = states.length
                ? states.map((state) => `• ${state.channelId}: <t:${Math.floor((state.lastIndexedTimestamp || 0) / 1000)}:R>`)
                : ["• Nenhum canal indexado ainda."];

            await interaction.editReply(
                `Mensagens: **${stats.messages}**\nChunks: **${stats.chunks}**\nCanais: **${stats.channels}**\n\n${lines.join("\n")}`
            );
            return;
        }

        if (subcommand === "clear") {
            DiscordMemoryService.clearAll();
            await interaction.editReply("Memória local apagada.");
            return;
        }

        if (subcommand === "repair") {
            DiscordMemoryService.repairIndexes();
            await interaction.editReply("Índices reparados.");
            return;
        }

        if (subcommand === "backfill_channel") {
            const channel = interaction.options.getChannel("channel", true) as TextChannel | ThreadChannel;
            const limit = interaction.options.getInteger("limit") ?? 1000;
            const count = await DiscordBackfillService.backfillChannel(channel, interaction, limit);
            await interaction.editReply(`Indexação concluída. ${count} mensagens processadas em ${channel.name}.`);
            return;
        }

        if (subcommand === "backfill_category") {
            const category = interaction.options.getChannel("category", true) as CategoryChannel;
            const limit = interaction.options.getInteger("limit_per_channel") ?? 500;
            const result = await DiscordBackfillService.backfillCategory(category, interaction, limit);
            await interaction.editReply(
                `Indexação concluída. ${result.messages} mensagens processadas em ${result.channels} canais.`
            );
        }
    },
};

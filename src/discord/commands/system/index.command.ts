import {
    CategoryChannel,
    ChannelType,
    ChatInputCommandInteraction,
    ContainerBuilder,
    MessageFlags,
    SlashCommandBuilder,
    TextChannel,
    TextDisplayBuilder,
    ThreadChannel,
} from "discord.js";
import { buildMemoryStatusContainer } from "@/discord/commands/shared/buildMemoryStatusContainer";
import { buildGuildCompletenessContainer } from "@/discord/commands/shared/buildGuildCompletenessContainer";
import { buildRuntimeStorageContainer } from "@/discord/commands/shared/buildRuntimeStorageContainer";
import { DiscordBackfillService } from "@/discord/live/DiscordBackfillService";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { RuntimeStorageService } from "@/runtime/storage/RuntimeStorageService";
import { SecurityService } from "@/security/SecurityService";
import { EMOJIS } from "@/discord/constants";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("index")
        .setDescription("Manage local Discord retrieval data")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addSubcommand((subcommand) =>
            subcommand.setName("status").setDescription("Show current local retrieval status")
        )
        .addSubcommand((subcommand) =>
            subcommand.setName("clear").setDescription("Reset runtime DBs, checkpoints, traces, and indexed messages")
        )
        .addSubcommand((subcommand) =>
            subcommand.setName("repair").setDescription("Reconnect and validate the local runtime storage")
        )
        .addSubcommand((subcommand) =>
            subcommand
                .setName("backfill_channel")
                .setDescription("Index channel history")
                .addChannelOption((option) =>
                    option
                        .setName("channel")
                        .setDescription("Channel to index")
                        .addChannelTypes(ChannelType.GuildText, ChannelType.PublicThread, ChannelType.PrivateThread)
                        .setRequired(true)
                )
                .addIntegerOption((option) =>
                    option
                        .setName("limit")
                        .setDescription("Maximum messages")
                        .setRequired(false)
                        .setMinValue(1)
                        .setMaxValue(5000)
                )
        )
        .addSubcommand((subcommand) =>
            subcommand
                .setName("backfill_category")
                .setDescription("Index all channels in a category")
                .addChannelOption((option) =>
                    option
                        .setName("category")
                        .setDescription("Category to index")
                        .addChannelTypes(ChannelType.GuildCategory)
                        .setRequired(true)
                )
                .addIntegerOption((option) =>
                    option
                        .setName("limit_per_channel")
                        .setDescription("Maximum messages per channel")
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

        await interaction.deferReply();
        const subcommand = interaction.options.getSubcommand();

        if (subcommand === "status") {
            const stats = await DiscordMemoryService.getStatsAsync();
            const states = await DiscordMemoryService.getIndexStateAsync();
            const runtimeStatus = RuntimeStorageService.getStatus();
            const completeness = await DiscordGuildDiscoveryService.getGuildCompletenessSummary(
                interaction.guild
            );
            await interaction.editReply({
                components: [
                    buildMemoryStatusContainer(stats, states, 10),
                    buildGuildCompletenessContainer(completeness),
                    buildRuntimeStorageContainer(runtimeStatus),
                ],
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        if (subcommand === "clear") {
            await RuntimeStorageService.resetAllRuntimeData();
            await DiscordMemoryService.resetRuntimeState();
            await interaction.editReply({
                components: [
                    new ContainerBuilder()
                        .setAccentColor(0xed4245)
                        .addTextDisplayComponents(
                            new TextDisplayBuilder().setContent("## Runtime state reset"),
                            new TextDisplayBuilder().setContent(
                                "Operational DB, checkpoint DB, traces, logs, and indexed Discord cache were deleted. The runtime will recreate clean storage on the next access."
                            )
                        ),
                    buildRuntimeStorageContainer(RuntimeStorageService.getStatus()),
                ],
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        if (subcommand === "repair") {
            await DiscordMemoryService.resetRuntimeState();
            await DiscordMemoryService.repairIndexesAsync();
            const stats = await DiscordMemoryService.getStatsAsync();
            const states = await DiscordMemoryService.getIndexStateAsync();
            await interaction.editReply({
                components: [
                    new ContainerBuilder()
                        .setAccentColor(0x57f287)
                        .addTextDisplayComponents(
                            new TextDisplayBuilder().setContent("## Runtime storage checked"),
                            new TextDisplayBuilder().setContent(
                                "Local runtime storage was revalidated and snapshots were reloaded."
                            )
                        ),
                    buildMemoryStatusContainer(stats, states, 10),
                    buildRuntimeStorageContainer(RuntimeStorageService.getStatus()),
                ],
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        if (subcommand === "backfill_channel") {
            const channel = interaction.options.getChannel("channel", true) as TextChannel | ThreadChannel;
            const limit = interaction.options.getInteger("limit") ?? 1000;
            const count = await DiscordBackfillService.backfillChannel(channel, interaction, limit);
            await interaction.editReply(`Indexing complete. ${count} messages processed in ${channel.name}.`);
            return;
        }

        if (subcommand === "backfill_category") {
            const category = interaction.options.getChannel("category", true) as CategoryChannel;
            const limit = interaction.options.getInteger("limit_per_channel") ?? 500;
            const result = await DiscordBackfillService.backfillCategory(category, interaction, limit);
            await interaction.editReply(
                `Indexing complete. ${result.messages} messages processed across ${result.channels} channels.`
            );
        }
    },
};

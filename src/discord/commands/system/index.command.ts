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
import { DiscordBackfillCrawler } from "@/discord/live/DiscordBackfillCrawler";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { RuntimeStorageService } from "@/runtime/storage/RuntimeStorageService";
import { SecurityService } from "@/security/SecurityService";
import { EMOJIS } from "@/discord/constants";
import type { BotClient } from "@/shared/appTypes";

function formatDuration(ms: number | null): string {
    if (ms == null || !Number.isFinite(ms)) return "—";
    const s = Math.max(0, Math.floor(ms / 1000));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m`;
    const h = Math.floor(m / 60);
    return `${h}h${m % 60}m`;
}

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
                .setDescription("Index channel history (omit limit to run a resumable background crawl)")
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
                        .setDescription("Maximum messages (omit for full resumable background crawl)")
                        .setRequired(false)
                        .setMinValue(1)
                        .setMaxValue(50000)
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
                        .setDescription("Maximum messages per channel (omit for full resumable background crawl)")
                        .setRequired(false)
                        .setMinValue(1)
                        .setMaxValue(5000)
                )
        )
        .addSubcommand((subcommand) =>
            subcommand.setName("crawl_status").setDescription("Show the background crawler queue and progress")
        )
        .addSubcommand((subcommand) =>
            subcommand
                .setName("crawl_stop")
                .setDescription("Stop the background crawl for a specific channel")
                .addChannelOption((option) =>
                    option
                        .setName("channel")
                        .setDescription("Channel whose background crawl should stop")
                        .addChannelTypes(ChannelType.GuildText, ChannelType.PublicThread, ChannelType.PrivateThread)
                        .setRequired(true)
                )
        )
        .addSubcommand((subcommand) =>
            subcommand.setName("crawl_pause").setDescription("Pause the background crawler")
        )
        .addSubcommand((subcommand) =>
            subcommand.setName("crawl_resume").setDescription("Resume the background crawler")
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
            const limit = interaction.options.getInteger("limit");
            if (limit == null) {
                await DiscordBackfillCrawler.enqueue(channel.id, {
                    reason: `/index backfill_channel by ${interaction.user.id}`,
                    priority: 1,
                    guildId: interaction.guildId,
                });
                await interaction.editReply(
                    `${EMOJIS.sync} Enqueued resumable background crawl for **${channel.name}**. Use \`/index crawl_status\` to follow progress.`
                );
                return;
            }
            const count = await DiscordBackfillService.backfillChannel(channel, interaction, limit);
            await interaction.editReply(`Indexing complete. ${count} messages processed in ${channel.name}.`);
            return;
        }

        if (subcommand === "backfill_category") {
            const category = interaction.options.getChannel("category", true) as CategoryChannel;
            const limit = interaction.options.getInteger("limit_per_channel");
            if (limit == null) {
                const children = category.children.cache.filter(
                    (child) => child.type === ChannelType.GuildText
                );
                let enqueued = 0;
                for (const child of children.values()) {
                    await DiscordBackfillCrawler.enqueue(child.id, {
                        reason: `/index backfill_category by ${interaction.user.id}`,
                        priority: 1,
                        guildId: interaction.guildId,
                    });
                    enqueued += 1;
                }
                await interaction.editReply(
                    `${EMOJIS.sync} Enqueued ${enqueued} channel(s) from **${category.name}** for resumable background crawl. Use \`/index crawl_status\` to follow progress.`
                );
                return;
            }
            const result = await DiscordBackfillService.backfillCategory(category, interaction, limit);
            await interaction.editReply(
                `Indexing complete. ${result.messages} messages processed across ${result.channels} channels.`
            );
            return;
        }

        if (subcommand === "crawl_status") {
            const queue = await DiscordBackfillCrawler.listQueue();
            const running = DiscordBackfillCrawler.isRunning();
            const paused = DiscordBackfillCrawler.isPaused();

            if (!queue.length) {
                await interaction.editReply(
                    `${EMOJIS.sync} Crawler ${running ? "running" : "stopped"}${paused ? " (paused)" : ""}. Queue is empty.`
                );
                return;
            }

            const lines: string[] = [];
            for (const job of queue.slice(0, 20)) {
                const crawlStates = await DiscordMemoryService.getChannelCrawlStateAsync(job.channelId);
                const crawlState = crawlStates[0];
                const oldest = crawlState?.oldestFetchedMessageId ?? "-";
                const age = formatDuration(Date.now() - job.enqueuedAt);
                lines.push(
                    `• <#${job.channelId}> · ${job.state} · ingested=${job.messagesIngested} · oldest=${oldest} · age=${age}${job.lastError ? ` · err: ${job.lastError}` : ""}`
                );
            }

            await interaction.editReply({
                components: [
                    new ContainerBuilder()
                        .setAccentColor(0x5865f2)
                        .addTextDisplayComponents(
                            new TextDisplayBuilder().setContent(
                                `## Crawler ${running ? "running" : "stopped"}${paused ? " (paused)" : ""}`
                            ),
                            new TextDisplayBuilder().setContent(lines.join("\n")),
                        ),
                ],
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        if (subcommand === "crawl_stop") {
            const channel = interaction.options.getChannel("channel", true) as TextChannel | ThreadChannel;
            await DiscordBackfillCrawler.stopChannel(channel.id);
            await interaction.editReply(`${EMOJIS.sync} Stopped background crawl for **${channel.name}**.`);
            return;
        }

        if (subcommand === "crawl_pause") {
            DiscordBackfillCrawler.pause();
            await interaction.editReply(`${EMOJIS.sync} Crawler paused.`);
            return;
        }

        if (subcommand === "crawl_resume") {
            DiscordBackfillCrawler.resume();
            await interaction.editReply(`${EMOJIS.sync} Crawler resumed.`);
            return;
        }
    },
};

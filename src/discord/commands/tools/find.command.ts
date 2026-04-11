import {
    ChannelType,
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import { GroundedResultUIService } from "@/discord/ui/GroundedResultUIService";
import { UIService } from "@/discord/ui/UIService";
import { EMOJIS } from "@/discord/constants";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

export = {
    data: new SlashCommandBuilder()
        .setName("find")
        .setDescription("Executa a busca especializada do Discord")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addStringOption((option) =>
            option
                .setName("topic")
                .setDescription("Tópico ou pergunta para buscar")
                .setRequired(true)
        )
        .addChannelOption((option) =>
            option
                .setName("channel")
                .setDescription("Opcional: limitar a busca a um canal")
                .addChannelTypes(ChannelType.GuildText, ChannelType.PublicThread, ChannelType.PrivateThread)
                .setRequired(false)
        )
        .addStringOption((option) =>
            option
                .setName("target")
                .setDescription("Opcional: id ou nome de canal/categoria")
                .setRequired(false)
        )
        .addStringOption((option) =>
            option
                .setName("author")
                .setDescription("Opcional: id, mention ou nome do autor")
                .setRequired(false)
        )
        .addBooleanOption((option) =>
            option
                .setName("ephemeral")
                .setDescription("Somente você pode ver o resultado")
                .setRequired(false)
        ),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        try {
            if (!SecurityService.isAdmin(interaction.user.id)) {
                await interaction.reply({
                    content: `${EMOJIS.error} Este comando está disponível apenas para administradores.`,
                    flags: MessageFlags.Ephemeral,
                });
                return;
            }

            const topic = interaction.options.getString("topic", true);
            const channel = interaction.options.getChannel("channel");
            const target = interaction.options.getString("target");
            const author = interaction.options.getString("author");
            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;

            await interaction.deferReply({
                flags: ephemeral ? MessageFlags.Ephemeral : undefined,
            });

            const resolvedAuthor = author
                ? await DiscordLiveService.resolveMemberIdentity(interaction.guild, author)
                : null;
            const resolvedTargets =
                !channel && target
                    ? await DiscordGuildDiscoveryService.resolveChannelTargets(
                          interaction.guild,
                          target,
                          interaction.channelId
                      )
                    : null;

            const retrieval = await UnifiedMessageRetrieval.retrieve({
                guild: interaction.guild,
                question: topic,
                currentChannelId: interaction.channelId,
                channelIds: channel
                    ? [channel.id]
                    : resolvedTargets?.resolvedIds.length
                      ? resolvedTargets.resolvedIds
                      : undefined,
                authorId: resolvedAuthor?.resolvedId || undefined,
                limit: 12,
            });

            const scopeSummary = [
                resolvedAuthor
                    ? resolvedAuthor.isCurrentGuildMember
                        ? `autor: ${resolvedAuthor.displayName}`
                        : `autor histórico: ${resolvedAuthor.displayName}`
                    : null,
                channel
                    ? `canal: #${channel.name}`
                    : resolvedTargets?.entries.length
                      ? `alvo: ${resolvedTargets.entries.map((entry) => entry.name).join(", ")}`
                      : null,
            ]
                .filter(Boolean)
                .join(" · ");

            const searchResults = retrieval.combinedResults;

            if (!searchResults.length) {
                await interaction.editReply(
                    UIService.formatStatusMessage(
                        EMOJIS.warning,
                        retrieval.liveEscalated
                            ? scopeSummary
                                ? `Nenhuma evidência foi encontrada para ${scopeSummary}, mesmo após atualizar o histórico do Discord.`
                                : "Nenhuma evidência foi encontrada, mesmo após atualizar o histórico do Discord."
                            : scopeSummary
                              ? `Nenhuma evidência armazenada foi encontrada para ${scopeSummary}.`
                              : "Nenhuma evidência armazenada foi encontrada.",
                        false
                    )
                );
                return;
            }

            await GroundedResultUIService.showSearchResults(
                interaction,
                retrieval.liveEscalated
                    ? scopeSummary
                        ? `Resultados do Discord (${scopeSummary})`
                        : "Resultados do Discord (cache + atualização ao vivo)"
                    : scopeSummary
                      ? `Resultados da memória do Discord (${scopeSummary})`
                      : "Resultados da memória do Discord",
                topic,
                searchResults
            );
        } catch (error) {
            console.error("Error in find command:", error);
            await interaction.editReply(
                UIService.formatStatusMessage(
                    EMOJIS.error,
                    "Ocorreu um erro durante a busca.",
                    false
                )
            );
        }
    },
};


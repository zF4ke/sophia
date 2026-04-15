import {
    CategoryChannel,
    ChannelType,
    ChatInputCommandInteraction,
    ContainerBuilder,
    MessageFlags,
    SlashCommandBuilder,
    TextDisplayBuilder,
} from "discord.js";
import { ProtectedChannelsService } from "@/app/ProtectedChannelsService";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

const PROTECTED_CHANNEL_TYPES = new Set([
    ChannelType.GuildText,
    ChannelType.GuildVoice,
    ChannelType.GuildAnnouncement,
    ChannelType.GuildForum,
    ChannelType.GuildStageVoice,
    ChannelType.GuildCategory,
    ChannelType.PublicThread,
    ChannelType.PrivateThread,
    ChannelType.AnnouncementThread,
]);

export = {
    data: new SlashCommandBuilder()
        .setName("superchannels")
        .setDescription("Manage protected channels (immune to all destructive tool calls)")
        .setContexts(0)
        .setIntegrationTypes(0)
        .setDMPermission(false)
        .addSubcommand((sub) =>
            sub
                .setName("list")
                .setDescription("List all channels with IDs (paste to configure protection)")
        )
        .addSubcommand((sub) =>
            sub
                .setName("status")
                .setDescription("Show currently protected channels")
        )
        .addSubcommand((sub) =>
            sub
                .setName("protect_all")
                .setDescription("Protect every channel currently in the server")
        )
        .addSubcommand((sub) =>
            sub
                .setName("unprotect")
                .setDescription("Remove protection from a specific channel")
                .addStringOption((opt) =>
                    opt
                        .setName("channel_id")
                        .setDescription("Channel ID to unprotect")
                        .setRequired(true)
                )
        )
        .addSubcommand((sub) =>
            sub
                .setName("protect")
                .setDescription("Add protection to a specific channel")
                .addStringOption((opt) =>
                    opt
                        .setName("channel_id")
                        .setDescription("Channel ID to protect")
                        .setRequired(true)
                )
        )
        .addSubcommand((sub) =>
            sub
                .setName("clear")
                .setDescription("Remove protection from ALL channels")
        ),

    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        await SecurityService.initialize();
        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ Apenas administradores podem gerir supercanais.",
                            });
            return;
        }

        const sub = interaction.options.getSubcommand();
        const slowSubcommands = new Set(["list", "status", "protect_all"]);
        if (slowSubcommands.has(sub)) {
            await interaction.deferReply();
        }

        // ── list ──────────────────────────────────────────────────────────
        if (sub === "list") {
            const guild = interaction.guild;
            if (!guild) {
                await interaction.editReply({ content: "❌ Sem contexto de servidor." });
                return;
            }

            await guild.channels.fetch();
            const protected_ = ProtectedChannelsService.getAll();

            const categories = guild.channels.cache
                .filter((c): c is CategoryChannel => c.type === ChannelType.GuildCategory)
                .sort((a, b) => a.position - b.position);

            const lines: string[] = ["### Todos os canais do servidor", ""];

            for (const [, category] of categories) {
                lines.push(`**📁 ${category.name}** (\`${category.id}\`)${protected_.has(category.id) ? " 🔒" : ""}`);
                const children = guild.channels.cache
                    .filter((c) => "parentId" in c && (c as { parentId?: string }).parentId === category.id)
                    .sort((a, b) => ("position" in a && "position" in b ? (a as { position: number }).position - (b as { position: number }).position : 0));

                for (const [, ch] of children) {
                    const lock = protected_.has(ch.id) ? " 🔒" : "";
                    lines.push(`  \\↳ #${ch.name} — \`${ch.id}\`${lock}`);
                }
            }

            // Uncategorised
            const uncategorised = guild.channels.cache.filter(
                (c) => !("parentId" in c && c.parentId) && c.type !== ChannelType.GuildCategory && PROTECTED_CHANNEL_TYPES.has(c.type),
            );
            if (uncategorised.size > 0) {
                lines.push("", "**Sem categoria**");
                for (const [, ch] of uncategorised) {
                    const lock = protected_.has(ch.id) ? " 🔒" : "";
                    lines.push(`  #${ch.name} — \`${ch.id}\`${lock}`);
                }
            }

            const content = lines.join("\n");
            const container = new ContainerBuilder()
                .setAccentColor(0x5865f2)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(content),
                    new TextDisplayBuilder().setContent(
                        `🔒 = protegido · ${protected_.size} canal(ais) protegido(s) · Use \`/superchannels protect_all\` para proteger tudo`
                    ),
                );

            await interaction.editReply({
                components: [container],
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        // ── status ────────────────────────────────────────────────────────
        if (sub === "status") {
            const protected_ = ProtectedChannelsService.getAll();
            const guild = interaction.guild;

            if (protected_.size === 0) {
                await interaction.editReply({
                    content: "Nenhum canal protegido. Use `/superchannels protect_all` ou `/superchannels protect`.",
                                    });
                return;
            }

            const lines = [`### 🔒 Canais protegidos (${protected_.size})`, ""];
            for (const id of protected_) {
                const ch = guild?.channels.cache.get(id);
                const name = ch ? `#${ch.name}` : `canal removido`;
                lines.push(`- ${name} — \`${id}\``);
            }

            const container = new ContainerBuilder()
                .setAccentColor(0x57f287)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(lines.join("\n")),
                );

            await interaction.editReply({
                components: [container],
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        // ── protect_all ───────────────────────────────────────────────────
        if (sub === "protect_all") {
            const guild = interaction.guild;
            if (!guild) {
                await interaction.editReply({ content: "❌ Sem contexto de servidor." });
                return;
            }

            await guild.channels.fetch();
            const allIds = guild.channels.cache
                .filter((c) => PROTECTED_CHANNEL_TYPES.has(c.type))
                .map((c) => c.id);

            ProtectedChannelsService.add(allIds);

            const container = new ContainerBuilder()
                .setAccentColor(0x57f287)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(
                        `### 🔒 ${allIds.length} canais protegidos\nTodas as ferramentas destrutivas estão bloqueadas para estes canais.`
                    ),
                );

            await interaction.editReply({
                components: [container],
                flags: MessageFlags.IsComponentsV2,
            });
            return;
        }

        // ── protect ───────────────────────────────────────────────────────
        if (sub === "protect") {
            const channelId = interaction.options.getString("channel_id", true).trim();
            ProtectedChannelsService.add([channelId]);

            const ch = interaction.guild?.channels.cache.get(channelId);
            const name = ch ? `#${ch.name}` : channelId;

            await interaction.reply({
                content: `🔒 ${name} está agora protegido.`,
                            });
            return;
        }

        // ── unprotect ─────────────────────────────────────────────────────
        if (sub === "unprotect") {
            const channelId = interaction.options.getString("channel_id", true).trim();
            ProtectedChannelsService.remove([channelId]);

            const ch = interaction.guild?.channels.cache.get(channelId);
            const name = ch ? `#${ch.name}` : channelId;

            await interaction.reply({
                content: `🔓 ${name} já não está protegido.`,
                            });
            return;
        }

        // ── clear ─────────────────────────────────────────────────────────
        if (sub === "clear") {
            ProtectedChannelsService.clear();
            await interaction.reply({
                content: "🔓 Proteção removida de todos os canais.",
                            });
            return;
        }
    },
};

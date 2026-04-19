import {
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
        .setDescription("Gerir canais protegidos (imunes a ações destrutivas)")
        .setContexts(0)
        .setIntegrationTypes(0)
        .setDMPermission(false)
        .addSubcommand((sub) =>
            sub
                .setName("status")
                .setDescription("Mostrar canais protegidos")
        )
        .addSubcommand((sub) =>
            sub
                .setName("protect_all")
                .setDescription("Proteger todos os canais do servidor")
        )
        .addSubcommand((sub) =>
            sub
                .setName("unprotect")
                .setDescription("Remover proteção de um canal")
                .addChannelOption((opt) =>
                    opt
                        .setName("canal")
                        .setDescription("Canal a desproteger")
                        .setRequired(true)
                )
        )
        .addSubcommand((sub) =>
            sub
                .setName("protect")
                .setDescription("Proteger um canal")
                .addChannelOption((opt) =>
                    opt
                        .setName("canal")
                        .setDescription("Canal a proteger")
                        .setRequired(true)
                )
        )
        .addSubcommand((sub) =>
            sub
                .setName("clear")
                .setDescription("Remover proteção de todos os canais")
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
        const slowSubcommands = new Set(["status", "protect_all"]);
        if (slowSubcommands.has(sub)) {
            await interaction.deferReply();
        }

        // ── status ────────────────────────────────────────────────────────
        if (sub === "status") {
            const protected_ = ProtectedChannelsService.getAll();
            const guild = interaction.guild;

            if (protected_.size === 0) {
                await interaction.editReply({
                    content: "Nenhum canal protegido. Usa `/superchannels protect_all` ou `/superchannels protect`.",
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
            const channelId = interaction.options.getChannel("canal", true).id;
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
            const channelId = interaction.options.getChannel("canal", true).id;
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

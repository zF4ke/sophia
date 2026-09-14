import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { loadAccessPanelData } from "./panelData";
import { buildAccessPanel } from "./panelRenderer";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";
import { SettingsService } from "@/app/SettingsService";
import { DISCORD_TOOL_NAMES } from "@/shared/discordTools";

export = {
    data: new SlashCommandBuilder()
        .setName("access")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0)
        .setDescription("Painel de administração para segurança do bot")
        .addBooleanOption(o => o.setName("enable_here").setDescription("Ativar ou desativar Sophia neste servidor"))
        .addBooleanOption(o => o.setName("enable_dms").setDescription("Permitir mensagens diretas para utilizadores com autorização global"))
        .addUserOption(o => o.setName("user").setDescription("Utilizador a autorizar ou remover"))
        .addRoleOption(o => o.setName("role").setDescription("Cargo a autorizar ou remover neste servidor"))
        .addStringOption(o => o.setName("level").setDescription("Ações permitidas").addChoices(
            { name: "Leitura", value: "read" }, { name: "Alterações", value: "write" }, { name: "Destrutivas", value: "destructive" }))
        .addStringOption(o => o.setName("mode").setDescription("Aprovação de alterações").addChoices(
            { name: "Pedir permissão", value: "ask" }, { name: "Autoaprovar", value: "auto" }))
        .addBooleanOption(o => o.setName("global").setDescription("Aplicar a autorização do utilizador em todos os locais ativos"))
        .addStringOption(o => o.setName("expires_at").setDescription("Data ISO 8601 com fuso horário; never remove a validade"))
        .addStringOption(o => o.setName("capability").setDescription("Regra para uma ferramenta, por exemplo send_message"))
        .addStringOption(o => o.setName("rule_tier").setDescription("Regra para um nível de ações").addChoices(
            { name: "Leitura", value: "read" }, { name: "Alterações", value: "write" }, { name: "Destrutivas", value: "destructive" }))
        .addStringOption(o => o.setName("rule").setDescription("Decisão dentro da autorização existente").addChoices(
            { name: "Permitir", value: "allow" }, { name: "Pedir permissão", value: "ask" }, { name: "Negar", value: "deny" }, { name: "Remover regra", value: "remove" }))
        .addBooleanOption(o => o.setName("remove").setDescription("Remover esta autorização"))
        .addBooleanOption(o => o.setName("ephemeral").setDescription("Mostrar só para ti")),
    async execute(interaction: ChatInputCommandInteraction, client: BotClient) {
        const responseFlags = interaction.options.getBoolean("ephemeral") ? MessageFlags.Ephemeral : undefined;
        await SecurityService.initialize();

        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ Você não tem permissão para usar este comando.",
                flags: responseFlags,
            });
            return;
        }

        const enabled = interaction.options.getBoolean("enable_here");
        const directMessages = interaction.options.getBoolean("enable_dms");
        const user = interaction.options.getUser("user");
        const role = interaction.options.getRole("role");
        const global = interaction.options.getBoolean("global") ?? false;
        const remove = interaction.options.getBoolean("remove") ?? false;
        const level = interaction.options.getString("level") as "read" | "write" | "destructive" | null;
        const mode = interaction.options.getString("mode") as "ask" | "auto" | null;
        const expiry = interaction.options.getString("expires_at");
        const tool = interaction.options.getString("capability");
        const tier = interaction.options.getString("rule_tier") as "read" | "write" | "destructive" | null;
        const rule = interaction.options.getString("rule") as "allow" | "ask" | "deny" | "remove" | null;
        if (tool || tier || rule) {
            if ((Boolean(user) === Boolean(role)) || (role && global) || (!interaction.guildId && !global) || !rule || (!tool && !tier) ||
                (tool && !DISCORD_TOOL_NAMES.includes(tool as never)) || level || mode || expiry || remove || enabled !== null || directMessages !== null) {
                await interaction.reply({ content: "Escolhe um utilizador ou cargo, capability ou rule_tier, e rule. Configura autorizações e disponibilidade numa operação separada. Cargos precisam deste servidor; em DMs usa global:true.", flags: responseFlags });
                return;
            }
            const access = structuredClone(SettingsService.load().access);
            const target = { subject: user ? "user" as const : "role" as const, subjectId: user?.id ?? role!.id,
                guildId: global ? undefined : interaction.guildId!, tool: tool ?? undefined, tier: tier ?? undefined };
            access.rules = (access.rules ?? []).filter(item => !(item.subject === target.subject && item.subjectId === target.subjectId &&
                item.guildId === target.guildId && item.tool === target.tool && item.tier === target.tier));
            if (rule !== "remove") access.rules.push({ ...target, decision: rule });
            SettingsService.update({ access });
            await interaction.reply({ content: `Regra ${rule === "remove" ? "removida" : "atualizada"}: ${target.subjectId} · ${tool ?? tier} · ${rule}. A regra não aumenta o nível autorizado. Leituras permitidas continuam sem confirmação.`, flags: responseFlags, allowedMentions: { parse: [] } });
            return;
        }
        if (expiry && expiry !== "never" && (!/^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(expiry) || !Number.isFinite(Date.parse(expiry)) || Date.parse(expiry) <= Date.now())) {
            await interaction.reply({ content: "Usa uma data futura com fuso horário, por exemplo 2027-01-01T12:00:00Z, ou never.", flags: responseFlags });
            return;
        }
        const expiresAt = expiry && expiry !== "never" ? new Date(expiry).toISOString() : undefined;
        if ((user && role) || (role && global) || ((!user && !role) && (global || remove || level || mode || expiry))) {
            await interaction.reply({ content: "Escolhe um utilizador ou um cargo. A autorização global aplica-se apenas a utilizadores.", flags: responseFlags });
            return;
        }
        if (enabled !== null || directMessages !== null || user || role) {
            if (!interaction.guildId && (enabled !== null || role || (user && !global))) {
                await interaction.reply({ content: "Em mensagens diretas, usa global:true para autorizar um utilizador. Servidores e cargos são configurados dentro do servidor.", flags: responseFlags });
                return;
            }
            const current = SettingsService.load();
            const access = structuredClone(current.access);
            if (directMessages !== null) access.directMessages = directMessages;
            const guildId = global ? undefined : interaction.guildId ?? undefined;
            if (user) {
                const previous = access.users.find(g => g.userId === user.id && g.guildId === guildId);
                access.users = access.users.filter(g => g.userId !== user.id || g.guildId !== guildId);
                if (!remove) access.users.push({ userId: user.id, guildId, level: level ?? previous?.level ?? "read", mode: mode ?? previous?.mode ?? "ask", expiresAt: expiry === null ? previous?.expiresAt : expiresAt });
            }
            if (role && interaction.guildId) {
                const previous = access.roles.find(g => g.roleId === role.id && g.guildId === interaction.guildId);
                access.roles = access.roles.filter(g => g.roleId !== role.id || g.guildId !== interaction.guildId);
                if (!remove) access.roles.push({ roleId: role.id, guildId: interaction.guildId, level: level ?? previous?.level ?? "read", mode: mode ?? previous?.mode ?? "ask", expiresAt: expiry === null ? previous?.expiresAt : expiresAt });
            }
            const guilds = new Set(current.guildAllowlist);
            if (enabled === true && interaction.guildId) guilds.add(interaction.guildId);
            if (enabled === false && interaction.guildId) guilds.delete(interaction.guildId);
            SettingsService.update({ access, guildAllowlist: [...guilds] });
            const target = user ? `<@${user.id}>${global ? " (autorização global)" : ""}` : role ? `<@&${role.id}>` : "disponibilidade";
            const location = interaction.guildId ? ` Sophia está ${guilds.has(interaction.guildId) ? "ativa" : "desativada"} no servidor ${interaction.guildId}.` : "";
            await interaction.reply({ content: `Acesso atualizado: ${target}.${location} Mensagens diretas: ${access.directMessages ? "ativas para utilizadores autorizados" : "desativadas"}.`, flags: responseFlags, allowedMentions: { parse: [] } });
            return;
        }

        await interaction.deferReply({ flags: responseFlags });

        const payload = await buildAccessPanel(client, await loadAccessPanelData(client), {
            view: "overview",
        });

        await interaction.editReply({
            ...payload,
            flags: MessageFlags.IsComponentsV2,
        });
    },
};

import { AttachmentBuilder, MessageFlags, SlashCommandBuilder, type ChatInputCommandInteraction } from "discord.js";
import { SkillStore } from "@/memory/SkillStore";
import { SecurityService } from "@/security/SecurityService";
import { readableDerived } from "@/security/DerivedSources";

export = {
    data: new SlashCommandBuilder().setName("skills").setDescription("Consulta procedimentos guardados e rascunhos")
        .addStringOption(option => option.setName("query").setDescription("Nome ou tema a procurar"))
        .addStringOption(option => option.setName("id").setDescription("ID do procedimento para consultar"))
        .addBooleanOption(option => option.setName("quarantine").setDescription("Operadores: consultar procedimentos sem proprietário ou origem definida"))
        .addBooleanOption(option => option.setName("adopt").setDescription("Operadores: assumir o procedimento revisto como rascunho privado neste canal"))
        .addIntegerOption(option => option.setName("revision").setDescription("Revisão exata do procedimento revisto").setMinValue(1))
        .addBooleanOption(option => option.setName("ephemeral").setDescription("Mostrar só para ti")),
    async execute(interaction: ChatInputCommandInteraction) {
        const privateResponse = interaction.options.getBoolean("ephemeral") === true;
        await interaction.deferReply({ flags: privateResponse ? MessageFlags.Ephemeral : undefined });
        const audience = { actorId: interaction.user.id, guildId: interaction.guildId, channelId: interaction.channelId };
        const id = interaction.options.getString("id");
        const quarantine = interaction.options.getBoolean("quarantine") === true;
        const adopt = interaction.options.getBoolean("adopt") === true;
        if (quarantine || adopt) {
            await SecurityService.initialize();
            if (!SecurityService.isAdmin(interaction.user.id)) { await interaction.editReply({ content: "Só operadores podem rever procedimentos em quarentena." }); return; }
        }
        try {
            if (adopt) {
                const revision = interaction.options.getInteger("revision");
                if (!id || revision === null) { await interaction.editReply({ content: "Consulta primeiro o procedimento e indica o seu id e revision para o assumir." }); return; }
                await SkillStore.adoptQuarantined(audience, id, revision);
                await interaction.editReply({ content: `Procedimento ${id}, revisão ${revision + 1}: assumido por ti como rascunho privado neste canal.`, allowedMentions: { parse: [] } });
                return;
            }
            const records = quarantine ? await SkillStore.quarantine(interaction.guildId, id ?? undefined) : id ? [await SkillStore.load(audience, id)].filter(skill => skill !== null) : await SkillStore.search(audience, interaction.options.getString("query") ?? "");
            const eligible = quarantine ? records : await readableDerived({ guild: interaction.guild, client: interaction.client,
                actorId: interaction.user.id, currentChannelId: interaction.channelId, privateResponse, question: "Inspect saved skills" }, records, skill => skill.sourceLinks ?? []);
            const selected = id ? eligible.filter(skill => skill.id === id) : eligible;
            if (!selected.length) { await interaction.editReply({ content: "Não encontrei procedimentos disponíveis com esses critérios." }); return; }
            const text = selected.map(skill => `${skill.name}\nID: ${skill.id} · Revisão: ${skill.revision} · ${skill.status === "ready" ? "Pronto" : "Rascunho"}\n${skill.description}`).join("\n\n");
            await interaction.editReply({ content: id ? `Procedimento ${id}: detalhes no ficheiro.` : `${selected.length} procedimento(s) disponível(is).`,
                files: [new AttachmentBuilder(Buffer.from(id ? JSON.stringify(selected[0], null, 2) : text), { name: id ? "procedimento.json" : "procedimentos.txt" })], allowedMentions: { parse: [] } });
        } catch (error) {
            await interaction.editReply({ content: error instanceof Error ? error.message : "Não consegui consultar os procedimentos.", allowedMentions: { parse: [] } });
        }
    },
};

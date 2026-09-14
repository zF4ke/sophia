import { AttachmentBuilder, MessageFlags, SlashCommandBuilder, type ChatInputCommandInteraction } from "discord.js";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { SecurityService } from "@/security/SecurityService";
import { readableDerived } from "@/security/DerivedSources";

export = {
    data: new SlashCommandBuilder().setName("memories").setDescription("Consulta memórias e revê importações antigas")
        .addStringOption(o => o.setName("query").setDescription("Tema a procurar"))
        .addBooleanOption(o => o.setName("quarantine").setDescription("Operadores: consultar importações sem origem confirmada"))
        .addIntegerOption(o => o.setName("page").setDescription("Página da quarentena").setMinValue(1))
        .addStringOption(o => o.setName("adopt").setDescription("Operadores: ID revisto a guardar como memória privada"))
        .addIntegerOption(o => o.setName("revision").setDescription("Revisão exata da memória revista").setMinValue(1))
        .addBooleanOption(option => option.setName("ephemeral").setDescription("Mostrar só para ti")),
    async execute(interaction: ChatInputCommandInteraction) {
        const privateResponse = interaction.options.getBoolean("ephemeral") === true;
        await interaction.deferReply({ flags: privateResponse ? MessageFlags.Ephemeral : undefined });
        const audience = { actorId: interaction.user.id, guildId: interaction.guildId, channelId: interaction.channelId, privateResponse };
        const quarantine = interaction.options.getBoolean("quarantine") === true;
        const adopt = interaction.options.getString("adopt");
        try {
            if (quarantine || adopt) {
                await SecurityService.initialize();
                if (!SecurityService.isAdmin(interaction.user.id)) throw new Error("Só operadores podem rever memórias em quarentena.");
            }
            if (adopt) {
                const revision = interaction.options.getInteger("revision");
                if (revision === null) throw new Error("Consulta primeiro a memória e indica a revisão exata.");
                await knowledgeStore.adoptQuarantined(audience, adopt, revision);
                await interaction.editReply({ content: `Memória ${adopt}, revisão ${revision + 1}: guardada como memória privada tua.`, allowedMentions: { parse: [] } });
                return;
            }
            const records = quarantine
                ? await knowledgeStore.quarantine(interaction.guildId, ((interaction.options.getInteger("page") ?? 1) - 1) * 50)
                : await readableDerived({ ...audience, guild: interaction.guild, client: interaction.client, currentChannelId: interaction.channelId, question: "Inspect memories" },
                    await knowledgeStore.search(audience, interaction.options.getString("query") ?? "", 50), memory => memory.sources);
            await interaction.editReply({ content: records.length ? `${records.length} memórias no ficheiro. ${quarantine ? "Cada página contém até 50 registos." : "Usa query para procurar por tema."}` : "Não encontrei memórias disponíveis com esses critérios.",
                files: records.length ? [new AttachmentBuilder(Buffer.from(JSON.stringify(records, null, 2)), { name: "memorias.json" })] : [], allowedMentions: { parse: [] } });
        } catch (error) {
            await interaction.editReply({ content: error instanceof Error ? error.message : "Não consegui consultar as memórias.", allowedMentions: { parse: [] } });
        }
    },
};

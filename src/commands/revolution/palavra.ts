import { SlashCommandBuilder, PermissionFlagsBits, ChatInputCommandInteraction, MessageFlags } from "discord.js";
import dedent from "dedent";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("palavra")
        .setDescription("🏴‍☠️")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1),
    async execute(interaction: ChatInputCommandInteraction) {
        const message = dedent`
            **“As sementes da injustiça implantada germinam um natural e súbito senso de revolta. É em nome da justiça que uma revolução floresce, uma das poucas colheitas que o semeador não quer colher.”**
            
            👉 [Junte-se à revolução](https://discord.com/oauth2/authorize?client_id=954890139665063986)
            ### Comandos:
            - \`/r\` para mostrar a bandeira da revolução. 🏴‍☠️
            - \`/encriptar <mensagem>\` para encriptar uma mensagem contra o ditador.
            - \`/decriptar <mensagem>\` para decriptar uma mensagem.
            - \`/palavra\` para espalhar a palavra da revolução. 🏴‍☠️
            
            ⚠️ Corra, antes que a censura chegue.
        `
        
        await interaction.reply({
            content: message,
            flags: MessageFlags.SuppressEmbeds
        });
    },
};

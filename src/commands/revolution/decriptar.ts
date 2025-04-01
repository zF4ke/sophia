import { SlashCommandBuilder, PermissionFlagsBits, ChatInputCommandInteraction, MessageFlags } from "discord.js";
import { decode, encode } from "@leodog896/vigenere-cipher"

const REVOLUTION_PASSWORD = "vivalarevoluciontrwzfzjfcubjnasvznkaziirtooaoebakmybfbhubendmwgmpnruiwpapurepzkwzasccirlotaepnugrvkgebypyubdeimsatgsibwfxkbnibwrmfokwptinhwcxtqzzefxczccozxbhxpfktdkzkodwqqfqrwmhxrimhgqtkblpbszwuefkcvctyvxsmhwkqrlrpwuxjfclwexuirramsqcrbrflgtuhzkcvhetjzdviibnqzienpjepkljgoo";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("decriptar")
        .setDescription("🏴‍☠️ Decriptar mensagens")
        .addStringOption(option =>
            option.setName("mensagem")
                .setDescription("Mensagem a ser decriptada")
                .setRequired(true))
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1),
    async execute(interaction: ChatInputCommandInteraction) {        
        if (interaction.user.id === "619503488057212958") {
            return await interaction.reply({
                content: "***Ditadores não são bem vindos na revolução.***\n\nApenas os verdadeiros revolucionários podem usar este comando. 🏴‍☠️",
                flags: MessageFlags.SuppressEmbeds
            });
        }

        const encryptedMessage = interaction.options.getString("mensagem", true);
        // use vigenere cipher to encrypt the message
        const message = decode(encryptedMessage, REVOLUTION_PASSWORD); 
        
        await interaction.reply({
            content: `Mensagem decriptada: ${message}`,
            flags: MessageFlags.Ephemeral
        });
    },
};

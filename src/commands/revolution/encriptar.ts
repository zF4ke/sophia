import { SlashCommandBuilder, PermissionFlagsBits, ChatInputCommandInteraction, MessageFlags } from "discord.js";
import { decode, encode } from "@leodog896/vigenere-cipher"

const REVOLUTION_PASSWORD = "vivalarevoluciontrwzfzjfcubjnasvznkaziirtooaoebakmybfbhubendmwgmpnruiwpapurepzkwzasccirlotaepnugrvkgebypyubdeimsatgsibwfxkbnibwrmfokwptinhwcxtqzzefxczccozxbhxpfktdkzkodwqqfqrwmhxrimhgqtkblpbszwuefkcvctyvxsmhwkqrlrpwuxjfclwexuirramsqcrbrflgtuhzkcvhetjzdviibnqzienpjepkljgoo";

module.exports = {
    data: new SlashCommandBuilder()
        .setName("encriptar")
        .setDescription("🏴‍☠️ Encriptar mensagens")
        .addStringOption(option =>
            option.setName("mensagem")
                .setDescription("Mensagem a ser encriptada")
                .setRequired(true))
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1),
    async execute(interaction: ChatInputCommandInteraction) {
        // if user has id 619503488057212958, return a message saying that the command is not available for him
        if (interaction.user.id === "619503488057212958") {
            return await interaction.reply({
                content: "***Ditadores não são bem vindos na revolução.***\n\nApenas os verdadeiros revolucionários podem usar este comando. 🏴‍☠️",
                flags: MessageFlags.SuppressEmbeds
            });
        }

        const message = interaction.options.getString("mensagem", true);
        // use vigenere cipher to encrypt the message
        const encryptedMessage = encode(message, REVOLUTION_PASSWORD); 
        
        await interaction.reply({
            content: `Mensagem encriptada: ${encryptedMessage}`,
            flags: MessageFlags.Ephemeral
        });
    },
};

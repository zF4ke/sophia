import { ChattingService } from "@/services/ai/ChattingService";
import { TextProcessingService } from "@/services/ai/TextProcessingService";
import { SecurityService } from "@/services/SecurityService";
import { UIService } from "@/services/UIService";
import dedent from "dedent";
import { Message, PermissionFlagsBits, TextChannel } from "discord.js";

module.exports = {
    name: "messageCreate",
    async execute(message: Message) {
        try {
            const channel = message.channel;
            if (!channel.isTextBased()) return;
            if (!(channel instanceof TextChannel)) return;

            if (message.author.id === message.client.user!.id) return; // Ignore messages from the bot itself

            // if it's a reply to a message from the bot, check if the user is an admin
            if (message.reference && message.reference.messageId) {
                const referencedMessage = await channel.messages.fetch(message?.reference?.messageId);
                if (referencedMessage.author.id === message.client.user!.id) {
                    const userId = message.author.id;
                    if (SecurityService.isAdmin(userId)) {
                        await talkReference(message, referencedMessage);
                    }
                }
                return;
            }

            const isMentioned = message.mentions.has(message.client.user!);
            if (isMentioned) {
                // check if it'a an admin with the SecurityService
                const userId = message.author.id;
                if (SecurityService.isAdmin(userId)) {
                    await talk(message);
                }
            }
            
        } catch (error) {
            console.error(error);
        }
    }
}

async function talk(message: Message) {
    try {
        // Check if the message is from a bot
        if (message.author.bot) return;

        const channel = message.channel;
        if (!channel.isTextBased()) return;
        if (!(channel instanceof TextChannel)) return;

        // Check if the bot has permission  to read messages in the channel
        const permissions = channel.permissionsFor(message.client.user!);
        if (!permissions) return;
        if (!permissions.has(PermissionFlagsBits.ViewChannel)) return;
        if (!permissions.has(PermissionFlagsBits.ReadMessageHistory)) return;
        if (!permissions.has(PermissionFlagsBits.SendMessages)) return;

        const prompt = message.content;
        const response = await ChattingService.generateChatResponse(
            channel,
            prompt,
            {
                userName: message.author.username,
            }
        );

        // Send the response back to the channel
        await UIService.sendLongMessage(message, response);
    } catch (error) {
        console.error(error);
    }
}

async function talkReference(message: Message, referencedMessage: Message) {
    try {
        // Check if the message is from a bot
        if (message.author.bot) return;

        const channel = message.channel;
        if (!channel.isTextBased()) return;
        if (!(channel instanceof TextChannel)) return;

        // Check if the bot has permission  to read messages in the channel
        const permissions = channel.permissionsFor(message.client.user!);
        if (!permissions) return;
        if (!permissions.has(PermissionFlagsBits.ViewChannel)) return;
        if (!permissions.has(PermissionFlagsBits.ReadMessageHistory)) return;
        if (!permissions.has(PermissionFlagsBits.SendMessages)) return;

        const additionalContext = dedent`
            ${message.author.username} está respondendo a uma mensagem que você enviou. Aqui está o que você disse: """
            ${TextProcessingService.removeMessageHeader(referencedMessage.content).trim()}
            """ Use isso para responder à mensagem dele.
        `;

        const prompt = message.content;
        const response = await ChattingService.generateChatResponse(
            channel,
            prompt,
            {
                userName: message.author.username,
                additionalContext,
            }
        );

        // Send the response back to the channel
        await UIService.sendLongMessage(message, response);
    } catch (error) {
        console.error(error);
    }
}
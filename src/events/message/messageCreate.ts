import { AgentOrchestrator } from "@/services/agent/AgentOrchestrator";
import { SecurityService } from "@/services/SecurityService";
import { DiscordMemoryService } from "@/services/memory/DiscordMemoryService";
import { UIService } from "@/services/UIService";
import dedent from "dedent";
import { Message, PermissionFlagsBits, TextChannel, ThreadChannel } from "discord.js";

export = {
    name: "messageCreate",
    async execute(message: Message) {
        try {
            const channel = message.channel;
            if (!channel.isTextBased()) return;
            if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) return;

            if (message.author.id === message.client.user!.id) return; // Ignore messages from the bot itself
            await DiscordMemoryService.ingestMessage(message);

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
        if (message.author.bot) return;

        const channel = message.channel;
        if (!channel.isTextBased()) return;
        if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) return;

        // Check if the bot has permission  to read messages in the channel
        const permissions = channel.permissionsFor(message.client.user!);
        if (!permissions) return;
        if (!permissions.has(PermissionFlagsBits.ViewChannel)) return;
        if (!permissions.has(PermissionFlagsBits.ReadMessageHistory)) return;
        if (!permissions.has(PermissionFlagsBits.SendMessages)) return;

        const prompt = message.content.replace(/<@!?[0-9]+>/g, "").trim();
        const response = await AgentOrchestrator.answerQuestion({
            question: prompt,
            user: message.author,
            guild: message.guild,
            currentChannelId: channel.id,
        });

        await UIService.sendLongMessage(message, UIService.formatAnswer(response.answer, response.citations));
    } catch (error) {
        console.error(error);
    }
}

async function talkReference(message: Message, referencedMessage: Message) {
    try {
        if (message.author.bot) return;

        const channel = message.channel;
        if (!channel.isTextBased()) return;
        if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) return;

        // Check if the bot has permission  to read messages in the channel
        const permissions = channel.permissionsFor(message.client.user!);
        if (!permissions) return;
        if (!permissions.has(PermissionFlagsBits.ViewChannel)) return;
        if (!permissions.has(PermissionFlagsBits.ReadMessageHistory)) return;
        if (!permissions.has(PermissionFlagsBits.SendMessages)) return;

        const additionalContext = dedent`
            ${message.author.username} está respondendo a uma mensagem que você enviou. Aqui está o que você disse: """
            ${referencedMessage.content.trim()}
            """ Use isso para responder à mensagem dele.
        `;

        const response = await AgentOrchestrator.answerQuestion({
            question: `${message.content}\n\n${additionalContext}`,
            user: message.author,
            guild: message.guild,
            currentChannelId: channel.id,
        });

        await UIService.sendLongMessage(message, UIService.formatAnswer(response.answer, response.citations));
    } catch (error) {
        console.error(error);
    }
}

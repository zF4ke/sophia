import { DebugService } from "@/discord/debug/DebugService";
import { ResponseActivityService } from "@/discord/responding/ResponseActivityIndicator";
import { UIService } from "@/discord/ui/UIService";
import { ConversationAdapter } from "@/discord/conversation/ConversationAdapter";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import { SecurityService } from "@/security/SecurityService";
import { Message, PermissionFlagsBits, TextChannel, ThreadChannel } from "discord.js";

export = {
    name: "messageCreate",
    async execute(message: Message) {
        try {
            const channel = message.channel;
            if (!channel.isTextBased()) return;
            if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) return;
            if (message.author.id === message.client.user!.id) return;

            await DiscordMemoryService.ingestMessage(message);

            if (message.reference?.messageId) {
                const referencedMessage = await message.fetchReference().catch(() => null);
                if (referencedMessage?.author.id === message.client.user!.id) {
                    if (SecurityService.isAdmin(message.author.id)) {
                        await respondToMessage(message, "reply");
                    }
                }
                return;
            }

            if (message.mentions.has(message.client.user!) && SecurityService.isAdmin(message.author.id)) {
                await respondToMessage(message, "mention");
            }
        } catch (error) {
            console.error(error);
        }
    },
};

async function respondToMessage(message: Message, trigger: "mention" | "reply") {
    let debugSession = null;
    let activityIndicator = null;

    try {
        if (message.author.bot) return;
        const channel = message.channel;
        if (!channel.isTextBased()) return;
        if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) return;

        const permissions = channel.permissionsFor(message.client.user!);
        if (!permissions) return;
        if (!permissions.has(PermissionFlagsBits.ViewChannel)) return;
        if (!permissions.has(PermissionFlagsBits.ReadMessageHistory)) return;
        if (!permissions.has(PermissionFlagsBits.SendMessages)) return;

        const prompt =
            trigger === "mention"
                ? message.content.replace(/<@!?[0-9]+>/g, "").trim()
                : message.content.trim();

        debugSession = await DebugService.startForMessage(message, prompt);
        activityIndicator = await ResponseActivityService.startForMessage(message, channel);
        await activityIndicator.startThinking();

        const input = await ConversationAdapter.fromMessage({
            message,
            trigger,
            question: prompt,
            debugSession,
            activityIndicator,
        });
        const result = await Runtime.answer(input);
        await activityIndicator.startTyping();
        const sentMessages = await UIService.sendLongMessage(
            message,
            UIService.formatAnswer(result.answer, result.citations)
        );
        await ConversationAdapter.bindResponseMessages({
            input,
            result,
            sentMessages,
        });
    } catch (error) {
        console.error(error);
        await debugSession?.finishError(error);
    } finally {
        await activityIndicator?.stop();
    }
}


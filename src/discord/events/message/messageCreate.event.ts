import { DebugService } from "@/discord/debug/DebugService";
import { ProgressStatusService } from "@/discord/responding/ProgressStatus";
import { ResponseActivityService } from "@/discord/responding/ResponseActivityIndicator";
import { UIService } from "@/discord/ui/UIService";
import { ConversationAdapter } from "@/discord/conversation/ConversationAdapter";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import { SecurityService } from "@/security/SecurityService";
import { isGuildAllowed } from "@/security/guildAllowlist";
import { ACCESS_POLICY_TARGETS } from "@/security/policyTargets";
import { Message, PermissionFlagsBits, TextChannel, ThreadChannel } from "discord.js";

export = {
    name: "messageCreate",
    async execute(message: Message) {
        try {
            const channel = message.channel;
            if (!channel.isTextBased()) return;
            if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) return;
            if (message.author.id === message.client.user!.id) return;
            if (!isGuildAllowed(message.guildId)) return;

            await SecurityService.initialize();

            DiscordMemoryService.ingestMessage(message).catch((err) => {
                console.warn("[messageCreate] ingestion failed (non-fatal):", (err as Error).message ?? err);
            });

            if (message.reference?.messageId) {
                const referencedMessage = await message.fetchReference().catch(() => null);
                if (referencedMessage?.author.id === message.client.user!.id) {
                    if (
                        await SecurityService.isTriggerEnabled(
                            ACCESS_POLICY_TARGETS.mention,
                            message.author.id,
                        ) &&
                        await SecurityService.isTriggerEnabled(
                            ACCESS_POLICY_TARGETS.reply,
                            message.author.id,
                        ) &&
                        await SecurityService.checkTriggerRateLimit(
                            message.author.id,
                            ACCESS_POLICY_TARGETS.reply,
                        )
                    ) {
                        await respondToMessage(message, "reply");
                    }
                }
                return;
            }

            if (
                message.mentions.has(message.client.user!) &&
                await SecurityService.isTriggerEnabled(
                    ACCESS_POLICY_TARGETS.mention,
                    message.author.id,
                ) &&
                await SecurityService.checkTriggerRateLimit(
                    message.author.id,
                    ACCESS_POLICY_TARGETS.mention,
                )
            ) {
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
    let progressStatus = null;

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
        progressStatus = ProgressStatusService.startForChannel(channel);
        await activityIndicator.startThinking();

        const input = await ConversationAdapter.fromMessage({
            message,
            trigger,
            question: prompt,
            debugSession,
            activityIndicator,
            progressNotifier: (summary: string) => progressStatus!.notify(summary),
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
        await progressStatus?.finalize();
        await activityIndicator?.stop();
    }
}

import { AccessPolicy } from "@/security/AccessPolicy";
import { DebugService } from "@/discord/debug/DebugService";
import { ProgressStatusService, type ProgressStatus } from "@/discord/responding/ProgressStatus";
import { ResponseActivityService, type ResponseActivityIndicator } from "@/discord/responding/ResponseActivityIndicator";
import { UIService } from "@/discord/ui/UIService";
import { ConversationAdapter } from "@/discord/conversation/ConversationAdapter";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import { SecurityService } from "@/security/SecurityService";
import { isGuildAllowed } from "@/security/guildAllowlist";
import { ACCESS_POLICY_TARGETS } from "@/security/policyTargets";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";
import { Message, PermissionFlagsBits, TextChannel, ThreadChannel } from "discord.js";

export = {
    name: "messageCreate",
    async execute(message: Message) {
        try {
            const channel = message.channel;
            if (!channel.isTextBased()) return;
            if (message.guild && !(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) return;
            if (message.author.id === message.client.user!.id) return;
            if (!isGuildAllowed(message.guildId)) return;

            await SecurityService.initialize();

            DiscordMemoryService.ingestMessage(message).catch((err) => {
                console.warn("[messageCreate] ingestion failed (non-fatal):", (err as Error).message ?? err);
            });

            if (message.author.bot || message.webhookId || await AccessPolicy.decide(message.author.id, message.guild, "none") === "deny") return;

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
                    return;
                }
            }

            if (
                (!message.guild || message.mentions.has(message.client.user!)) &&
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
    const releaseRequest = ActiveRequestTracker.begin();
    let debugSession = null;
    let activityIndicator: ResponseActivityIndicator | null = null;
    let progressStatus: ProgressStatus | null = null;

    try {
        if (message.author.bot) return;
        const channel = message.channel;
        if (!channel.isTextBased()) return;
        if (message.guild && !(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) return;

        if (channel instanceof TextChannel || channel instanceof ThreadChannel) {
            const permissions = channel.permissionsFor(message.client.user!);
            if (!permissions?.has(PermissionFlagsBits.ViewChannel) || !permissions.has(PermissionFlagsBits.ReadMessageHistory) || !permissions.has(channel instanceof ThreadChannel ? PermissionFlagsBits.SendMessagesInThreads : PermissionFlagsBits.SendMessages)) return;
        }

        const prompt =
            trigger === "mention"
                ? message.content.split(`<@${message.client.user!.id}>`).join("").split(`<@!${message.client.user!.id}>`).join("").trim()
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
            progressNotifier: async (summary: string) => { await activityIndicator?.stop(); await progressStatus!.notify(summary); },
        });
        input.onTaskBound = taskId => progressStatus?.bindTask?.(taskId);
        const result = await Runtime.answer(input);
        await activityIndicator.startTyping();
        const formatted = UIService.formatAnswer(result.answer, result.citations);
        // An empty send reads as being ignored. The runtime already guarantees
        // a non-empty answer; this guard catches anything that slips past it.
        const sentMessages = await UIService.sendLongMessage(
            message,
            formatted.trim() ? formatted : "Não consegui gerar uma resposta desta vez. Tenta de novo.",
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
        releaseRequest();
    }
}

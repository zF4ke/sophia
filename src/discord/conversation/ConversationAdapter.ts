import { sourceReference } from "@/shared/sourceReference";
import { AccessPolicy } from "@/security/AccessPolicy";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { SettingsService } from "@/app/SettingsService";
import { interactionApprovalTransport } from "@/discord/approval/InteractionApprovalTransport";
import {
    ChatInputCommandInteraction,
    Message,
    ThreadChannel,
} from "discord.js";
import { createApprovalGate, createBatchApprovalGate, createProtectedBlockNotifier } from "@/discord/approval/ApprovalGate";
import { buildConversationContext, buildReplyContext } from "@/discord/conversation/ConversationIdentity";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { RuntimeAnswer, RuntimeDebugSession, TurnInput, TurnTrigger } from "@/runtime/contracts";

function getNativeThreadId(channel: Message["channel"] | ChatInputCommandInteraction["channel"]): string | null {
    if (channel instanceof ThreadChannel) {
        return channel.id;
    }
    return null;
}

function getInteractionDisplayName(interaction: ChatInputCommandInteraction): string {
    const member = interaction.member;
    if (member && typeof member === "object" && "displayName" in member && typeof member.displayName === "string") {
        return member.displayName;
    }
    return interaction.user.displayName || interaction.user.username;
}

function getMessageDisplayName(message: Message): string {
    return message.member?.displayName || message.author.displayName || message.author.username;
}

export class ConversationAdapter {
    public static async fromInteraction(options: {
        interaction: ChatInputCommandInteraction;
        question: string;
        debugSession?: RuntimeDebugSession | null;
        progressNotifier?: ((summary: string) => Promise<void>) | null;
    }): Promise<TurnInput> {
        const { interaction, question, debugSession } = options;
        const approvalTransport = interactionApprovalTransport(interaction);
        return {
            question,
            responseVisibility: interaction.ephemeral === false ? "channel" : "private",
            allowDreaming: true,
            attachments: (() => { const file = interaction.options?.getAttachment?.("attachment"); return file ? [{ id: file.id, name: file.name, url: file.url, contentType: file.contentType, size: file.size }] : []; })(),
            user: interaction.user,
            authorize: AccessPolicy.forActor(interaction.user.id, interaction.guild),
            requesterDisplayName: getInteractionDisplayName(interaction),
            guild: interaction.guild,
            currentChannelId: interaction.channelId,
            nativeThreadId: getNativeThreadId(interaction.channel),
            debugSession,
            requestedWebMode: "auto",
            trigger: "talk",
            replyContext: null,
            referencedMessage: null,
            approvalGate: approvalTransport
                ? createApprovalGate(approvalTransport)
                : undefined,
            batchApprovalGate: approvalTransport
                ? createBatchApprovalGate(approvalTransport)
                : undefined,
            protectedBlockNotifier: approvalTransport
                ? createProtectedBlockNotifier(approvalTransport)
                : undefined,
            progressNotifier: options.progressNotifier ?? null,
            conversation: await buildConversationContext({
                guild: interaction.guild,
                currentChannelId: interaction.channelId,
                nativeThreadId: getNativeThreadId(interaction.channel),
                trigger: "talk",
            }),
        };
    }

    public static async fromMessage(options: {
        message: Message;
        trigger: TurnTrigger;
        question: string;
        debugSession?: RuntimeDebugSession | null;
        activityIndicator?: {
            startThinking(): Promise<void>;
            startTyping(): Promise<void>;
            stop(): Promise<void>;
        } | null;
        progressNotifier?: ((summary: string) => Promise<void>) | null;
    }): Promise<TurnInput> {
        const { message, trigger, question, debugSession, activityIndicator } = options;
        const replyContext = message.reference?.messageId ? await buildReplyContext(message) : null;
        const referencedMessage = message.reference?.messageId ? await message.fetchReference().catch(() => null) : null;
        return {
            question,
            allowDreaming: true,
            sourceMessageUrl: message.url,
            attachments: [...(message.attachments?.values() ?? [])].map(file => ({ id: file.id, name: file.name, url: file.url, contentType: file.contentType, size: file.size })),
            user: message.author,
            sourceMessageId: message.id,
            authorize: AccessPolicy.forActor(message.author.id, message.guild),
            requesterDisplayName: getMessageDisplayName(message),
            guild: message.guild,
            currentChannelId: message.channelId,
            nativeThreadId: getNativeThreadId(message.channel),
            debugSession,
            requestedWebMode: "auto",
            trigger,
            replyContext,
            referencedMessage,
            approvalGate: message.channel && "send" in message.channel
                ? createApprovalGate(message.channel as import("discord.js").SendableChannels)
                : undefined,
            batchApprovalGate: message.channel && "send" in message.channel
                ? createBatchApprovalGate(message.channel as import("discord.js").SendableChannels)
                : undefined,
            protectedBlockNotifier: message.channel && "send" in message.channel
                ? createProtectedBlockNotifier(message.channel as import("discord.js").SendableChannels)
                : undefined,
            progressNotifier: options.progressNotifier ?? null,
            activityIndicator: activityIndicator ?? null,
            conversation: await buildConversationContext({
                guild: message.guild,
                currentChannelId: message.channelId,
                nativeThreadId: getNativeThreadId(message.channel),
                trigger,
                message,
            }),
        };
    }

    public static async bindResponseMessages(options: {
        input: TurnInput;
        result: RuntimeAnswer;
        sentMessages: Message[];
    }): Promise<void> {
        if (!options.sentMessages.length) {
            return;
        }

        await DiscordMemoryService.recordConversationMessages({
            requestId: options.result.requestId,
            threadId: options.result.threadId,
            guildId: options.input.guild?.id || null,
            channelId: options.input.currentChannelId || null,
            messageIds: options.sentMessages.map((message) => message.id),
        });
        if (options.input.allowDreaming && options.result.outcome === "completed" && SettingsService.load().memory.dreamingEnabled) {
            const references = await taskStore.requestSources(options.result.requestId);
            if (references === null) return;
            await knowledgeStore.enqueueDream(options.result.requestId, {
                audience: { actorId: options.input.user.id, guildId: options.input.guild?.id ?? null, channelId: options.input.currentChannelId ?? null, privateResponse: options.input.responseVisibility === "private" || !options.input.guild },
                question: options.input.question.slice(0, 8000), answer: options.result.answer.slice(0, 12000),
                taskId: options.result.taskId,
                toolEvidence: options.result.toolRuns.slice(-20).map(run => ({ tool: run.tool, summary: run.summary.slice(0, 1500), succeeded: !run.errorMessage })),
                sources: [...new Set([...options.input.sourceMessageUrl ? [options.input.sourceMessageUrl] : [], ...options.sentMessages.map(message => message.url),
                    ...references.map(source => sourceReference(source))])],
            }).catch(error => console.error("[Dreaming] Could not enqueue delivered conversation", error));
        }
    }
}

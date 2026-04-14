import {
    ChatInputCommandInteraction,
    Message,
    ThreadChannel,
} from "discord.js";
import { createApprovalGate } from "@/discord/approval/ApprovalGate";
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
    }): Promise<TurnInput> {
        const { interaction, question, debugSession } = options;
        return {
            question,
            user: interaction.user,
            requesterDisplayName: getInteractionDisplayName(interaction),
            guild: interaction.guild,
            currentChannelId: interaction.channelId,
            nativeThreadId: getNativeThreadId(interaction.channel),
            debugSession,
            requestedWebMode: "auto",
            trigger: "talk",
            replyContext: null,
            referencedMessage: null,
            approvalGate: interaction.channel && "send" in interaction.channel
                ? createApprovalGate(interaction.channel as import("discord.js").SendableChannels)
                : undefined,
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
    }): Promise<TurnInput> {
        const { message, trigger, question, debugSession, activityIndicator } = options;
        const replyContext = trigger === "reply" ? await buildReplyContext(message) : null;
        const referencedMessage = trigger === "reply" ? await message.fetchReference().catch(() => null) : null;
        return {
            question,
            user: message.author,
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
    }
}

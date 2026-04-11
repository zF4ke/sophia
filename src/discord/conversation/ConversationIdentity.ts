import type { Guild, Message } from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { ConversationContext, ReplyContext, TurnTrigger } from "@/runtime/contracts";

const MAX_REPLY_DEPTH = 12;

function buildChannelConversationKey(guildId: string | null, channelId: string | null): string {
    return [guildId || "dm", channelId || "no-channel", "channel"].join(":");
}

function buildNativeThreadConversationKey(guildId: string | null, nativeThreadId: string): string {
    return [guildId || "dm", nativeThreadId, "native-thread"].join(":");
}

function buildReplyConversationKey(guildId: string | null, anchorMessageId: string): string {
    return [guildId || "dm", anchorMessageId, "reply-chain"].join(":");
}

export async function resolveReplyAnchor(message: Message): Promise<{
    anchorMessageId: string | null;
    referencedMessage: Message | null;
    conversationThreadId: string | null;
}> {
    if (!message.reference?.messageId || !message.channel.isTextBased()) {
        return {
            anchorMessageId: null,
            referencedMessage: null,
            conversationThreadId: null,
        };
    }

    let current = await message.fetchReference().catch(() => null);
    const referencedMessage = current;
    let anchorMessageId = current?.id || null;

    for (let depth = 0; current && depth < MAX_REPLY_DEPTH; depth += 1) {
        const storedThreadId = await DiscordMemoryService.resolveConversationThreadIdForMessage(
            current.id
        );
        if (storedThreadId) {
            return {
                anchorMessageId: current.id,
                referencedMessage,
                conversationThreadId: storedThreadId,
            };
        }

        if (!current.reference?.messageId) {
            break;
        }

        const next = await current.fetchReference().catch(() => null);
        if (!next) {
            break;
        }

        anchorMessageId = next.id;
        current = next;
    }

    return {
        anchorMessageId,
        referencedMessage,
        conversationThreadId: null,
    };
}

export async function buildConversationContext(options: {
    guild: Guild | null;
    currentChannelId?: string | null;
    nativeThreadId?: string | null;
    trigger: TurnTrigger;
    message?: Message | null;
}): Promise<ConversationContext> {
    const guildId = options.guild?.id || null;

    if (options.nativeThreadId) {
        return {
            key: buildNativeThreadConversationKey(guildId, options.nativeThreadId),
            kind: "native_thread",
            trigger: options.trigger,
            nativeThreadId: options.nativeThreadId,
            replyAnchorMessageId: null,
        };
    }

    if (options.message?.reference?.messageId) {
        const { anchorMessageId, conversationThreadId } = await resolveReplyAnchor(options.message);
        if (conversationThreadId) {
            return {
                key: conversationThreadId,
                kind: "reply_chain",
                trigger: options.trigger,
                replyAnchorMessageId: anchorMessageId,
                nativeThreadId: null,
            };
        }

        if (anchorMessageId) {
            return {
                key: buildReplyConversationKey(guildId, anchorMessageId),
                kind: "reply_chain",
                trigger: options.trigger,
                replyAnchorMessageId: anchorMessageId,
                nativeThreadId: null,
            };
        }
    }

    return {
        key: buildChannelConversationKey(guildId, options.currentChannelId || null),
        kind: "channel",
        trigger: options.trigger,
        replyAnchorMessageId: null,
        nativeThreadId: null,
    };
}

export async function buildReplyContext(message: Message): Promise<ReplyContext | null> {
    if (!message.reference?.messageId) {
        return null;
    }

    const referencedMessage = await message.fetchReference().catch(() => null);
    if (!referencedMessage) {
        return null;
    }

    return {
        messageId: referencedMessage.id,
        authorId: referencedMessage.author.id,
        authorName: referencedMessage.author.username,
        authorDisplayName: referencedMessage.member?.displayName || referencedMessage.author.displayName || referencedMessage.author.username,
        content: referencedMessage.content || "",
        jumpLink: referencedMessage.url,
    };
}

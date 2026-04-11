import { beforeEach, describe, expect, it, vi } from "vitest";
import { buildConversationContext } from "@/discord/conversation/ConversationIdentity";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

function createMessage(options: {
    id: string;
    reference?: { messageId: string } | null;
    referencedMessage?: any;
}) {
    return {
        id: options.id,
        reference: options.reference || null,
        channel: {
            isTextBased: () => true,
        },
        fetchReference: vi.fn().mockResolvedValue(options.referencedMessage || null),
    } as any;
}

describe("ConversationIdentity", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("reuses an existing stored thread id anywhere in a reply chain", async () => {
        const root = createMessage({ id: "m-root" });
        const botReply = createMessage({
            id: "m-bot",
            reference: { messageId: "m-root" },
            referencedMessage: root,
        });
        const userReply = createMessage({
            id: "m-user",
            reference: { messageId: "m-bot" },
            referencedMessage: botReply,
        });

        vi.spyOn(
            DiscordMemoryService,
            "resolveConversationThreadIdForMessage"
        ).mockImplementation(async (messageId: string) =>
            messageId === "m-root" ? "g1:c1:channel" : null
        );

        const context = await buildConversationContext({
            guild: { id: "g1" } as any,
            currentChannelId: "c1",
            trigger: "reply",
            message: userReply,
        });

        expect(context).toEqual({
            key: "g1:c1:channel",
            kind: "reply_chain",
            trigger: "reply",
            replyAnchorMessageId: "m-root",
            nativeThreadId: null,
        });
    });

    it("anchors nested replies at the oldest reachable message instead of nesting reply keys", async () => {
        const root = createMessage({ id: "m-root" });
        const firstReply = createMessage({
            id: "m-reply-1",
            reference: { messageId: "m-root" },
            referencedMessage: root,
        });
        const secondReply = createMessage({
            id: "m-reply-2",
            reference: { messageId: "m-reply-1" },
            referencedMessage: firstReply,
        });

        vi.spyOn(
            DiscordMemoryService,
            "resolveConversationThreadIdForMessage"
        ).mockResolvedValue(null);

        const firstContext = await buildConversationContext({
            guild: { id: "g1" } as any,
            currentChannelId: "c1",
            trigger: "reply",
            message: firstReply,
        });
        const secondContext = await buildConversationContext({
            guild: { id: "g1" } as any,
            currentChannelId: "c1",
            trigger: "reply",
            message: secondReply,
        });

        expect(firstContext.key).toBe("g1:m-root:reply-chain");
        expect(secondContext.key).toBe("g1:m-root:reply-chain");
        expect(secondContext.replyAnchorMessageId).toBe("m-root");
        expect(secondContext.key).not.toBe("g1:m-reply-1:reply-chain");
    });
});


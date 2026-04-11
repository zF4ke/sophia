import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationAdapter } from "@/discord/conversation/ConversationAdapter";
import * as ConversationIdentity from "@/discord/conversation/ConversationIdentity";

describe("ConversationAdapter", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("identifies who is talking to Sophia and who authored the replied message", async () => {
        vi.spyOn(ConversationIdentity, "buildConversationContext").mockResolvedValue({
            key: "g1:m-root:reply-chain",
            kind: "reply_chain",
            trigger: "reply",
            replyAnchorMessageId: "m-root",
            nativeThreadId: null,
        });

        vi.spyOn(ConversationIdentity, "buildReplyContext").mockResolvedValue({
            messageId: "m-referenced",
            authorId: "u-alice",
            authorName: "alice",
            authorDisplayName: "Alice",
            content: "I think we should delay the launch.",
            jumpLink: "https://discord.com/channels/g1/c1/m-referenced",
        });

        const referencedMessage = {
            id: "m-referenced",
            author: {
                id: "u-alice",
                username: "alice",
                displayName: "Alice",
            },
            member: {
                displayName: "Alice",
            },
            content: "I think we should delay the launch.",
            url: "https://discord.com/channels/g1/c1/m-referenced",
        };

        const message = {
            author: {
                id: "u-bob",
                username: "bob",
                displayName: "Bobby",
            },
            member: {
                displayName: "Bob",
            },
            guild: { id: "g1" },
            channelId: "c1",
            channel: {},
            reference: { messageId: "m-referenced" },
            fetchReference: vi.fn().mockResolvedValue(referencedMessage),
        } as any;

        const input = await ConversationAdapter.fromMessage({
            message,
            trigger: "reply",
            question: "what did she mean?",
        });

        expect(input.user.id).toBe("u-bob");
        expect(input.requesterDisplayName).toBe("Bob");
        expect(input.replyContext).toEqual({
            messageId: "m-referenced",
            authorId: "u-alice",
            authorName: "alice",
            authorDisplayName: "Alice",
            content: "I think we should delay the launch.",
            jumpLink: "https://discord.com/channels/g1/c1/m-referenced",
        });
        expect(input.referencedMessage).toBe(referencedMessage);
        expect(input.conversation).toEqual({
            key: "g1:m-root:reply-chain",
            kind: "reply_chain",
            trigger: "reply",
            replyAnchorMessageId: "m-root",
            nativeThreadId: null,
        });
    });
});

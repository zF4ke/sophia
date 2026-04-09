import { describe, expect, it } from "vitest";
import { ConversationService } from "@/services/ConversationService";

function createMessage(
    id: string,
    authorId: string,
    createdTimestamp: number,
    content: string,
    referenceMessageId?: string
) {
    return {
        id,
        author: {
            id: authorId,
            bot: false,
            username: `user-${authorId}`,
        },
        content,
        createdTimestamp,
        reference: referenceMessageId ? { messageId: referenceMessageId } : undefined,
    } as any;
}

describe("ConversationService", () => {
    it("groups chronological messages into conversations using forward time gaps", () => {
        const base = 1_700_000_000_000;
        const messages = [
            createMessage("1", "a", base, "primeira"),
            createMessage("2", "b", base + 60_000, "segunda"),
            createMessage("3", "a", base + 20 * 60_000, "terceira"),
        ];

        const grouped = ConversationService.groupMessagesByConversation(messages);
        expect(grouped).toHaveLength(2);
        expect(grouped[0].map((message) => message.id)).toEqual(["1", "2"]);
        expect(grouped[1].map((message) => message.id)).toEqual(["3"]);
    });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import { PermissionFlagsBits, TextChannel } from "discord.js";

vi.mock("@/discord/debug/DebugService", () => ({
    DebugService: {
        startForMessage: vi.fn(async () => ({ finishError: vi.fn() })),
    },
}));

vi.mock("@/discord/responding/ResponseActivityIndicator", () => ({
    ResponseActivityService: {
        startForMessage: vi.fn(async () => ({
            startThinking: vi.fn(),
            startTyping: vi.fn(),
            stop: vi.fn(),
        })),
    },
}));

vi.mock("@/discord/responding/ProgressStatus", () => ({
    ProgressStatusService: {
        startForChannel: vi.fn(() => ({
            notify: vi.fn(),
            finalize: vi.fn(),
        })),
    },
}));

vi.mock("@/discord/ui/UIService", () => ({
    UIService: {
        formatAnswer: vi.fn(() => "formatted"),
        sendLongMessage: vi.fn(async () => []),
    },
}));

vi.mock("@/discord/conversation/ConversationAdapter", () => ({
    ConversationAdapter: {
        fromMessage: vi.fn(async () => ({ question: "hi" })),
        bindResponseMessages: vi.fn(async () => undefined),
    },
}));

vi.mock("@/memory/DiscordMemoryService", () => ({
    DiscordMemoryService: {
        ingestMessage: vi.fn(() => Promise.resolve()),
    },
}));

vi.mock("@/runtime/Runtime", () => ({
    Runtime: {
        answer: vi.fn(async () => ({ answer: "ok", citations: [] })),
    },
}));

import event from "@/discord/events/message/messageCreate.event";
import { SecurityService } from "@/security/SecurityService";
import { Runtime } from "@/runtime/Runtime";

function createTextChannel() {
    return Object.assign(Object.create(TextChannel.prototype), {
        isTextBased: () => true,
        permissionsFor: vi.fn(() => ({
            has: vi.fn((permission: bigint) =>
                [
                    PermissionFlagsBits.ViewChannel,
                    PermissionFlagsBits.ReadMessageHistory,
                    PermissionFlagsBits.SendMessages,
                ].includes(permission)
            ),
        })),
    });
}

function createReplyMessage() {
    const channel = createTextChannel();
    return {
        author: { id: "user-1", bot: false },
        client: { user: { id: "bot-1" } },
        channel,
        content: "reply body",
        mentions: { has: vi.fn(() => false) },
        reference: { messageId: "ref-1" },
        fetchReference: vi.fn(async () => ({ author: { id: "bot-1" } })),
    } as any;
}

describe("messageCreate reply gating", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        vi.spyOn(SecurityService, "initialize").mockResolvedValue();
        vi.spyOn(SecurityService, "checkTriggerRateLimit").mockResolvedValue(true);
    });

    it("does not answer replies when pinging is off even if reply is enabled", async () => {
        const isTriggerEnabled = vi.spyOn(SecurityService, "isTriggerEnabled");
        isTriggerEnabled.mockImplementation(async (trigger) => trigger !== "trigger_mention");

        const message = createReplyMessage();
        await event.execute(message);

        expect(isTriggerEnabled).toHaveBeenCalledWith("trigger_mention", "user-1");
        expect(isTriggerEnabled).not.toHaveBeenCalledWith("trigger_reply", "user-1");
        expect(Runtime.answer).not.toHaveBeenCalled();
    });

    it("answers replies only when both pinging and reply are enabled", async () => {
        const isTriggerEnabled = vi.spyOn(SecurityService, "isTriggerEnabled").mockResolvedValue(true);

        const message = createReplyMessage();
        await event.execute(message);

        expect(isTriggerEnabled).toHaveBeenCalledWith("trigger_mention", "user-1");
        expect(isTriggerEnabled).toHaveBeenCalledWith("trigger_reply", "user-1");
        expect(SecurityService.checkTriggerRateLimit).toHaveBeenCalledWith("user-1", "trigger_reply");
        expect(Runtime.answer).toHaveBeenCalledTimes(1);
    });
});

import { Collection, TextChannel } from "discord.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import { handleDebugPanelInteraction } from "@/discord/debug/debugPanelInteractions";
import { DebugService } from "@/discord/debug/DebugService";
import { AgentOrchestrator } from "@/agent/AgentOrchestrator";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

describe("debug integration", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        DebugModeService.resetForTests();
    });

    it("enables and disables the global debug panel for admins", async () => {
        vi.spyOn(SecurityService, "initialize").mockResolvedValue(undefined);
        vi.spyOn(SecurityService, "isAdmin").mockReturnValue(true);

        const update = vi.fn().mockResolvedValue(undefined);
        const interaction = {
            customId: "debug:enable",
            user: { id: "admin" },
            update,
            reply: vi.fn().mockResolvedValue(undefined),
        } as any;

        expect(await handleDebugPanelInteraction(interaction)).toBe(true);
        expect(DebugModeService.isEnabled()).toBe(true);
        expect(update).toHaveBeenCalledTimes(1);
    });

    it("starts a public trace for slash-command flows when enabled", async () => {
        DebugModeService.setEnabled(true);

        const send = vi.fn().mockResolvedValue({
            edit: vi.fn().mockResolvedValue(undefined),
        });
        const channel = Object.assign(Object.create(TextChannel.prototype), {
            send,
        });
        const interaction = {
            channel,
        } as any;

        const session = await DebugService.startForInteraction(
            interaction,
            "Qual foi a decisão?"
        );

        expect(session).not.toBeNull();
        expect(send).toHaveBeenCalledTimes(1);
    });

    it("threads the debug session through mention-reply answer flows", async () => {
        const eventModuleImport = (await import("@/discord/events/message/messageCreate.event")) as any;
        const eventModule = eventModuleImport.default || eventModuleImport;
        vi.spyOn(SecurityService, "isAdmin").mockReturnValue(true);
        vi.spyOn(DiscordMemoryService, "ingestMessage").mockResolvedValue(undefined);
        vi.spyOn(DebugService, "startForMessage").mockResolvedValue({
            finishError: vi.fn().mockResolvedValue(undefined),
        } as any);
        vi.spyOn(AgentOrchestrator, "answerQuestion").mockResolvedValue({
            answer: "ok",
            citations: [],
            classification: {
                mode: "direct_answer",
                reason: "x",
            },
            toolRuns: [],
        });
        const sendLongMessage = vi
            .spyOn((await import("@/discord/ui/UIService")).UIService, "sendLongMessage")
            .mockResolvedValue(undefined);

        const channel = Object.assign(Object.create(TextChannel.prototype), {
            isTextBased: () => true,
            permissionsFor: () => ({
                has: () => true,
            }),
            id: "channel-1",
            messages: {
                fetch: vi.fn(),
            },
        });
        const message = {
            channel,
            author: { id: "admin-1", bot: false },
            client: { user: { id: "bot-1" } },
            guild: null,
            content: "<@bot-1> oi",
            mentions: { has: () => true },
            reference: null,
        } as any;

        await eventModule.execute(message, {
            commands: new Collection(),
        } as unknown as BotClient);

        expect(DebugService.startForMessage).toHaveBeenCalled();
        expect(AgentOrchestrator.answerQuestion).toHaveBeenCalledWith(
            expect.objectContaining({
                debugSession: expect.anything(),
            })
        );
        expect(sendLongMessage).toHaveBeenCalledTimes(1);
    });
});

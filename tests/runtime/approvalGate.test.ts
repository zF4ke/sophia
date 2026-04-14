import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway, type ToolChatResult } from "@/ai/ModelGateway";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import type { ApprovalRequest, ApprovalResult, TurnInput } from "@/runtime/contracts";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";

function createInput(overrides: Partial<TurnInput> = {}): TurnInput {
    return {
        question: "test",
        user: { id: "u-requester" } as any,
        requesterDisplayName: "Requester",
        guild: { id: "g1", name: "Test Guild" } as any,
        currentChannelId: "c1",
        nativeThreadId: null,
        requestedWebMode: "off",
        trigger: "talk",
        replyContext: null,
        referencedMessage: null,
        conversation: {
            key: "g1:c1:channel",
            kind: "channel",
            trigger: "talk",
            replyAnchorMessageId: null,
            nativeThreadId: null,
        },
        ...overrides,
    };
}

let toolCallCounter = 0;

function makeToolCallResult(calls: Array<{ name: string; args: Record<string, unknown> }>): ToolChatResult {
    return {
        content: null,
        toolCalls: calls.map((c) => ({
            id: `tc-${++toolCallCounter}`,
            type: "function" as const,
            function: { name: c.name, arguments: JSON.stringify(c.args) },
        })),
        finishReason: "tool_calls",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

function makeFinishResult(answer: string): ToolChatResult {
    return {
        content: null,
        toolCalls: [{
            id: `tc-${++toolCallCounter}`,
            type: "function" as const,
            function: { name: "finish", arguments: JSON.stringify({ answer }) },
        }],
        finishReason: "tool_calls",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

describe("approval gate", () => {
    beforeEach(async () => {
        vi.restoreAllMocks();
        toolCallCounter = 0;
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";

        vi.spyOn(DiscordMemoryService, "getRecentRuntimeRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "recordToolRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
    });

    it("calls approvalGate for write tools and proceeds when approved", async () => {
        const approvalGate = vi.fn<(req: ApprovalRequest) => Promise<ApprovalResult>>()
            .mockResolvedValue({ approved: true, decidedBy: "admin-1", decidedAt: Date.now() });

        // Mock the capability run so we don't hit real Discord
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "create_channel") {
                return {
                    ...cap,
                    run: vi.fn().mockResolvedValue({
                        tool: "create_channel",
                        summary: "Created text channel #new-channel (ch-new).",
                        data: { channelId: "ch-new", channelName: "new-channel", type: "text" },
                    }),
                };
            }
            return cap;
        });

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "create_channel", args: { name: "new-channel" } },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Channel created."));

        const result = await Runtime.answer(createInput({ approvalGate }));

        expect(approvalGate).toHaveBeenCalledOnce();
        const request = approvalGate.mock.calls[0][0];
        expect(request.toolName).toBe("create_channel");
        expect(request.sideEffectLevel).toBe("write");
        expect(result.answer).toBe("Channel created.");
    });

    it("injects denied result when approvalGate denies", async () => {
        const approvalGate = vi.fn<(req: ApprovalRequest) => Promise<ApprovalResult>>()
            .mockResolvedValue({ approved: false, decidedBy: "admin-1", decidedAt: Date.now() });

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "clear_messages", args: { channel_id: "c1", count: 5 } },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Action was denied."));

        const result = await Runtime.answer(createInput({ approvalGate }));

        expect(approvalGate).toHaveBeenCalledOnce();
        expect(result.answer).toBe("Action was denied.");
        // The tool should appear as denied in toolRuns
        expect(result.toolRuns.some((r) => r.tool === "clear_messages")).toBe(false);
    });

    it("auto-denies when no approvalGate is provided", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "create_channel", args: { name: "new-ch" } },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Could not create channel."));

        const result = await Runtime.answer(createInput({ approvalGate: undefined }));

        // Tool should not have been executed (no approval gate = auto-deny)
        expect(result.answer).toBe("Could not create channel.");
    });

    it("does NOT call approvalGate for read-only tools", async () => {
        const approvalGate = vi.fn<(req: ApprovalRequest) => Promise<ApprovalResult>>();

        vi.spyOn(DiscordLiveService, "getGuildContext").mockResolvedValue({
            name: "Test Guild",
            memberCount: 10,
            channelCount: 5,
            id: "g1",
        } as any);

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "get_guild_context", args: {} },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Here is the guild info."));

        const result = await Runtime.answer(createInput({ approvalGate }));

        expect(approvalGate).not.toHaveBeenCalled();
        expect(result.answer).toBe("Here is the guild info.");
    });

    it("calls approvalGate for destructive tools with sideEffectLevel destructive", async () => {
        const approvalGate = vi.fn<(req: ApprovalRequest) => Promise<ApprovalResult>>()
            .mockResolvedValue({ approved: true, decidedBy: "admin-1", decidedAt: Date.now() });

        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "clear_messages") {
                return {
                    ...cap,
                    run: vi.fn().mockResolvedValue({
                        tool: "clear_messages",
                        summary: "Deleted 10 message(s) from #general.",
                        data: { deletedCount: 10, channelId: "c1", channelName: "general" },
                    }),
                };
            }
            return cap;
        });

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "clear_messages", args: { channel_id: "c1", count: 10 } },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Messages cleared."));

        const result = await Runtime.answer(createInput({ approvalGate }));

        expect(approvalGate).toHaveBeenCalledOnce();
        const request = approvalGate.mock.calls[0][0];
        expect(request.sideEffectLevel).toBe("destructive");
        expect(result.answer).toBe("Messages cleared.");
    });

    it("does not spend the latency budget while waiting for approval", async () => {
        vi.useFakeTimers();

        const approvalGate = vi.fn<(req: ApprovalRequest) => Promise<ApprovalResult>>()
            .mockImplementation(
                () =>
                    new Promise<ApprovalResult>((resolve) => {
                        setTimeout(() => {
                            resolve({
                                approved: true,
                                decidedBy: "admin-1",
                                decidedAt: Date.now(),
                            });
                        }, 20_000);
                    }),
            );

        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "create_channel") {
                return {
                    ...cap,
                    run: vi.fn().mockResolvedValue({
                        tool: "create_channel",
                        summary: "Created text channel #slow-approved (ch-slow).",
                        data: {
                            channelId: "ch-slow",
                            channelName: "slow-approved",
                            type: "text",
                        },
                    }),
                };
            }
            return cap;
        });

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "create_channel", args: { name: "slow-approved" } },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Channel created after approval."));

        const resultPromise = Runtime.answer(createInput({ approvalGate }));
        await vi.advanceTimersByTimeAsync(20_000);
        const result = await resultPromise;

        expect(approvalGate).toHaveBeenCalledOnce();
        expect(result.answer).toBe("Channel created after approval.");
        expect(ModelGateway.generateWithTools).toHaveBeenCalledTimes(2);

        vi.useRealTimers();
    });
});

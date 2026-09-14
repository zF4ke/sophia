import path from "path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway, type ToolChatResult } from "@/ai/ModelGateway";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { SettingsService } from "@/app/SettingsService";
import { Runtime } from "@/runtime/Runtime";
import { taskStore } from "@/runtime/tasks/TaskStore";
import type { TurnInput } from "@/runtime/contracts";

function createInput(guild: unknown): TurnInput {
    return {
        question: "faz um artefacto com o dossiê",
        authorize: async () => "allow",
        user: { id: "u-requester" } as any,
        requesterDisplayName: "Requester",
        guild: guild as any,
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
    };
}

let toolCallCounter = 0;

function makeToolCallResult(name: string, args: Record<string, unknown>): ToolChatResult {
    return {
        content: null,
        toolCalls: [{
            id: `tc-${++toolCallCounter}`,
            type: "function" as const,
            function: { name, arguments: JSON.stringify(args) },
        }],
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

function makeGuildHarness() {
    const sentMessages: Array<{ id: string; url: string }> = [];
    const fakeChannel = {
        id: "c1",
        isTextBased: () => true,
        send: vi.fn(async (payload: unknown) => {
            const sent = { id: `m-${sentMessages.length + 1}`, url: `https://discord.com/channels/g1/c1/m-${sentMessages.length + 1}`, payload };
            sentMessages.push(sent);
            return sent;
        }),
    };
    const guild = {
        id: "g1",
        name: "Test",
        client: { user: { id: "bot-1" } },
        channels: { cache: new Map([["c1", fakeChannel]]) },
    };
    return { guild, fakeChannel, sentMessages };
}

describe("artifact guard", () => {
    // SettingsService.update persists to the (test-isolated) settings.json,
    // which other test files read. Snapshot the original and restore after
    // each test so a value like maxLatencyBudgetMs: 1 never leaks.
    const originalSettings = JSON.parse(JSON.stringify(SettingsService.load()));

    afterEach(() => {
        SettingsService.save(JSON.parse(JSON.stringify(originalSettings)));
    });

    beforeEach(async () => {
        vi.restoreAllMocks();
        toolCallCounter = 0;
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";

        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath: path.join(
                    process.cwd(),
                    "storage",
                    "test-runtime-store",
                    `artifact-guard-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });

        vi.spyOn(DiscordMemoryService, "getRecentRuntimeRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "recordToolRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
    });

    it("rejects finish after invalid artifact arguments and accepts once the card is fixed and sent", async () => {
        const { guild, sentMessages } = makeGuildHarness();
        const recordSpy = vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult("artifact_send", { title: 42, sections: [{ body: "x" }] }))
            .mockResolvedValueOnce(makeFinishResult("Pronto!"))
            .mockResolvedValueOnce(makeToolCallResult("artifact_send", {
                title: "Dossiê",
                sections: [{ heading: "A", body: "Corpo do dossiê." }],
            }))
            .mockResolvedValueOnce(makeFinishResult("Card enviado."));

        const result = await Runtime.answer(createInput(guild));

        expect(result.answer).toBe("Card enviado.");
        expect(sentMessages).toHaveLength(1);

        const persisted = recordSpy.mock.calls.at(-1)?.[0] as { traceEvents: Array<{ label: string; detail: string }> };
        const labels = persisted.traceEvents.map((t) => t.label);
        expect(labels).toContain("artifact_pending");
        expect(labels).toContain("artifact_guard");
        expect(labels).toContain("artifact_resolved");
    });

    it("lets finish through after the correction budget is exhausted", async () => {
        const { guild } = makeGuildHarness();
        const recordSpy = vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult("artifact_send", { title: 42, sections: [{ body: "x" }] }))
            .mockResolvedValueOnce(makeFinishResult("Primeira tentativa de sair."))
            .mockResolvedValueOnce(makeToolCallResult("artifact_send", { title: 42, sections: [] }))
            .mockResolvedValueOnce(makeFinishResult("Segunda tentativa de sair."))
            .mockResolvedValueOnce(makeFinishResult("O cartão falhou, não consegui enviar."));

        const result = await Runtime.answer(createInput(guild));

        expect(result.answer).toBe("O cartão falhou, não consegui enviar.");
        const persisted = recordSpy.mock.calls.at(-1)?.[0] as { traceEvents: Array<{ label: string; detail: string }> };
        const guardEvents = persisted.traceEvents.filter((t) => t.label === "artifact_guard");
        expect(guardEvents).toHaveLength(2);
    });

    it("pauses without another model call when an artifact send has an unknown outcome", async () => {
        const { guild, fakeChannel } = makeGuildHarness();
        fakeChannel.send.mockRejectedValueOnce(new Error("Connection lost after sending"));
        const model = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult("artifact_send", { title: "Dossiê", sections: [{ body: "Evidence" }] }));
        const result = await Runtime.answer(createInput(guild));
        expect(result.outcome).toBe("paused");
        expect(result.answer).toContain("não consegui confirmar");
        expect(model).toHaveBeenCalledOnce();
        expect(fakeChannel.send).toHaveBeenCalledOnce();
        expect((await taskStore.snapshot(result.taskId!, "u-requester", "c1", "g1"))?.actions)
            .toMatchObject([{ tool: "artifact_send", status: "unknown" }]);
    });
});

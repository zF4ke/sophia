import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway, type ToolChatResult } from "@/ai/ModelGateway";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import type { TurnInput } from "@/runtime/contracts";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";

function createInput(overrides: Partial<TurnInput> = {}): TurnInput {
    return {
        question: "test",
        user: { id: "u-requester" } as any,
        requesterDisplayName: "Requester",
        guild: { id: "g1", name: "Test" } as any,
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

function makeStartLongTaskResult(args: Record<string, unknown>): ToolChatResult {
    return {
        content: null,
        toolCalls: [{
            id: `tc-${++toolCallCounter}`,
            type: "function" as const,
            function: { name: "start_long_task", arguments: JSON.stringify(args) },
        }],
        finishReason: "tool_calls",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

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

function makeMalformedMarkupResult(content: string): ToolChatResult {
    return {
        content,
        toolCalls: [],
        finishReason: "stop",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

function makeLengthBlankResult(): ToolChatResult {
    return {
        content: null,
        toolCalls: [],
        finishReason: "length",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

describe("runtime stall correction", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        toolCallCounter = 0;
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";

        vi.spyOn(DiscordMemoryService, "getRecentRuntimeRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "recordToolRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "pruneExpiredThreadNotes").mockResolvedValue({ removed: 0 });
        // Default stall classifier: not a stall (tests override when needed)
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({ stall: false });
    });

    it("rejects a stalling finish answer and allows the model to retry", async () => {
        vi.spyOn(ModelGateway, "generateJson")
            .mockResolvedValueOnce({ stall: true })
            .mockResolvedValue({ stall: false });

        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            // First call: model tries to stall
            .mockResolvedValueOnce(makeFinishResult("Vou dar uma olhada agora!"))
            // Second call: model answers properly after correction
            .mockResolvedValueOnce(makeFinishResult("Não encontrei nenhuma mensagem sobre isso."));

        const result = await Runtime.answer(createInput({ question: "O que o João disse?" }));

        // The stall should have been rejected (first finish) and a real answer produced (second finish)
        expect(result.answer).toBe("Não encontrei nenhuma mensagem sobre isso.");
        // generateWithTools should have been called twice (stall rejection + retry)
        expect(generateSpy).toHaveBeenCalledTimes(2);
    });

    it("rejects a stalling finish twice before accepting the third attempt", async () => {
        vi.spyOn(ModelGateway, "generateJson")
            .mockResolvedValueOnce({ stall: true })
            .mockResolvedValueOnce({ stall: true })
            .mockResolvedValue({ stall: false });

        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeFinishResult("Vou dar uma olhada agora!"))
            .mockResolvedValueOnce(makeFinishResult("Te aviso quando terminar!"))
            .mockResolvedValueOnce(makeFinishResult("Não encontrei evidência sobre isso."));

        const result = await Runtime.answer(createInput({ question: "O que o João disse?" }));

        expect(result.answer).toBe("Não encontrei evidência sobre isso.");
        expect(generateSpy).toHaveBeenCalledTimes(3);
    });

    it("rejects a stalling finish twice before accepting the third attempt", async () => {
        vi.spyOn(ModelGateway, "generateJson")
            .mockResolvedValueOnce({ stall: true })
            .mockResolvedValueOnce({ stall: true })
            .mockResolvedValue({ stall: false });

        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeFinishResult("Vou dar uma olhada agora!"))
            .mockResolvedValueOnce(makeFinishResult("Te aviso quando terminar!"))
            .mockResolvedValueOnce(makeFinishResult("Não encontrei evidência sobre isso."));

        const result = await Runtime.answer(createInput({ question: "O que o João disse?" }));

        expect(result.answer).toBe("Não encontrei evidência sobre isso.");
        expect(generateSpy).toHaveBeenCalledTimes(3);
    });

    it("does not reject a non-stalling direct answer", async () => {
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeFinishResult("Olá! Como posso ajudar?"));

        const result = await Runtime.answer(createInput({ question: "Oi" }));

        expect(result.answer).toBe("Olá! Como posso ajudar?");
        expect(generateSpy).toHaveBeenCalledTimes(1);
    });

    it("rejects raw tool-call markup and asks the model to retry properly", async () => {
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeMalformedMarkupResult([
                "<minimax:tool_call>",
                "<invoke name=\"search__messages\">",
                "<parameter name=\"limit\">25</parameter>",
                "</invoke>",
                "</minimax:tool_call>",
            ].join("\n")))
            .mockResolvedValueOnce(makeFinishResult("Consegui continuar normalmente."));

        const result = await Runtime.answer(createInput({ question: "Continua a busca." }));

        expect(result.answer).toBe("Consegui continuar normalmente.");
        expect(generateSpy).toHaveBeenCalledTimes(2);
    });

    it("retries with stripped context when finishReason=length returns empty output", async () => {
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            // First call: model returns nothing due to context bloat
            .mockResolvedValueOnce(makeLengthBlankResult())
            // Second call: model answers with stripped context
            .mockResolvedValueOnce(makeFinishResult("Ta rindo demais mesmo kkk"));

        const result = await Runtime.answer(createInput({ question: "ta rindo demais viu" }));

        expect(result.answer).toBe("Ta rindo demais mesmo kkk");
        expect(generateSpy).toHaveBeenCalledTimes(2);
    });

    it("does not retry more than once on repeated finishReason=length", async () => {
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            // First call: empty
            .mockResolvedValueOnce(makeLengthBlankResult())
            // Second call: still empty after stripping
            .mockResolvedValueOnce(makeLengthBlankResult());

        const result = await Runtime.answer(createInput({ question: "ta rindo demais viu" }));

        // Should stop with no answer after one retry
        expect(result.answer).toBe("");
        expect(generateSpy).toHaveBeenCalledTimes(2);
    });
});

describe("runtime start_long_task interception", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        toolCallCounter = 0;
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";

        vi.spyOn(DiscordMemoryService, "getRecentRuntimeRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "recordToolRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "pruneExpiredThreadNotes").mockResolvedValue({ removed: 0 });
        // Default classifier: not a stall (long-task tests manage their own overrides)
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({ stall: false });
    });

    it("intercepts start_long_task without dispatching to capability layer", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            // First call: model calls start_long_task
            .mockResolvedValueOnce(makeStartLongTaskResult({
                reason: "bulk pagination",
                estimated_tool_calls: 50,
                estimated_seconds: 120,
            }))
            // Second call: model finishes
            .mockResolvedValueOnce(makeFinishResult("Done."));

        const result = await Runtime.answer(createInput({ question: "List everything" }));

        expect(result.answer).toBe("Done.");
    });

    it("rejects finish after start_long_task when no evidence was produced", async () => {
        // Override classifier: first two finishes are promises, third is real
        vi.spyOn(ModelGateway, "generateJson")
            .mockResolvedValueOnce({ stall: true })
            .mockResolvedValueOnce({ stall: true })
            .mockResolvedValue({ stall: false });

        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeStartLongTaskResult({
                reason: "bulk scan",
                estimated_tool_calls: 100,
                estimated_seconds: 200,
            }))
            // Model tries to finish with a descriptive but evidence-free answer
            .mockResolvedValueOnce(makeFinishResult("Já tenho os IDs necessários e vou buscar as mensagens agora."))
            // Second correction: still no evidence
            .mockResolvedValueOnce(makeFinishResult("Estou começando a coletar as mensagens do canal."))
            // Third attempt accepted (corrections exhausted)
            .mockResolvedValueOnce(makeFinishResult("Não consegui coletar as mensagens."));

        const result = await Runtime.answer(createInput({ question: "Faça uma varredura grande no canal" }));

        expect(result.answer).toBe("Não consegui coletar as mensagens.");
        // start_long_task + 3 finish attempts = 4 calls
        expect(generateSpy).toHaveBeenCalledTimes(4);
    });

    it("omitting estimates raises the budget to the hard cap", async () => {
        const recordSpy = vi.spyOn(DiscordMemoryService, "recordRuntimeRun")
            .mockResolvedValue(undefined);
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeStartLongTaskResult({ reason: "no estimates given" }))
            .mockResolvedValueOnce(makeFinishResult("Done."));

        const result = await Runtime.answer(createInput({ question: "Big task" }));

        expect(result.answer).toBe("Done.");
        const persistedArgs = recordSpy.mock.calls.at(-1)?.[0] as { traceEvents: Array<{ label: string; detail: string }> };
        const longTaskTrace = persistedArgs.traceEvents.find((t) => t.label === "long_task");
        expect(longTaskTrace?.detail).toMatch(/calls=200/);
        expect(longTaskTrace?.detail).toMatch(/latency=600000ms/);
    });

    it("second start_long_task call in same turn is idempotent", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeStartLongTaskResult({
                reason: "first",
                estimated_tool_calls: 50,
                estimated_seconds: 120,
            }))
            .mockResolvedValueOnce(makeStartLongTaskResult({
                reason: "second",
                estimated_tool_calls: 100,
                estimated_seconds: 300,
            }))
            .mockResolvedValueOnce(makeFinishResult("Done."));

        const result = await Runtime.answer(createInput({ question: "Big task" }));

        expect(result.answer).toBe("Done.");
    });

    it("rejects finish when a corpus-size request has not retrieved enough messages yet", async () => {
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        const retrieveRun = vi.fn()
            .mockResolvedValueOnce({
                tool: "retrieve_messages",
                summary: "ordered history evidence; 1000 history and 0 semantic result(s) from cached Discord history.",
                data: {
                    targetChannelIds: ["c1"],
                    mode: "history",
                    historyMessageCount: 1000,
                    accumulatedUniqueCount: 1000,
                    continuation: {
                        continuationAvailable: true,
                        history: {
                            continuationAvailable: true,
                            perChannelOldestMessageId: { c1: "m-oldest-1" },
                        },
                    },
                    exhaustion: { historyExhausted: false },
                },
            })
            .mockResolvedValueOnce({
                tool: "retrieve_messages",
                summary: "ordered history evidence; 20000 history and 0 semantic result(s) from cached Discord history.",
                data: {
                    targetChannelIds: ["c1"],
                    mode: "history",
                    historyMessageCount: 1000,
                    accumulatedUniqueCount: 20000,
                    continuation: {
                        continuationAvailable: false,
                        history: {
                            continuationAvailable: false,
                            perChannelOldestMessageId: { c1: null },
                        },
                    },
                    exhaustion: { historyExhausted: true },
                },
            });

        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "retrieve_messages") {
                return { ...cap, run: retrieveRun };
            }
            return cap;
        });

        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "retrieve_messages", args: { channelIds: ["c1"], mode: "history", order: "oldest" } },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Já deu para analisar.")) // should be rejected
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "retrieve_messages", args: { channelIds: ["c1"], mode: "history", order: "oldest", cursor: { history: { perChannelOldestMessageId: { c1: "m-oldest-1" }, continuationAvailable: true }, continuationAvailable: true } } },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Agora sim."));

        const result = await Runtime.answer(createInput({
            question: "Faça uma psico-analise com base em 20000 mensagens do chat do One Person.",
        }));

        expect(result.answer).toBe("Agora sim.");
        expect(generateSpy).toHaveBeenCalledTimes(4);
    });

    it("rejects finish for corpus-size requests when retrieve_messages was never called", async () => {
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeFinishResult("Vou fazer com base no que lembro."))
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "retrieve_messages", args: { channelIds: ["c1"], mode: "history" } },
            ]))
            .mockResolvedValueOnce(makeFinishResult("Agora com pesquisa."));

        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "retrieve_messages") {
                return {
                    ...cap,
                    run: vi.fn().mockResolvedValue({
                        tool: "retrieve_messages",
                        summary: "ordered history evidence; 50 history and 0 semantic result(s) from cached Discord history.",
                        data: {
                            targetChannelIds: ["c1"],
                            mode: "history",
                            historyMessageCount: 50,
                            accumulatedUniqueCount: 50,
                            continuation: { continuationAvailable: false, history: { continuationAvailable: false, perChannelOldestMessageId: { c1: null } } },
                            exhaustion: { historyExhausted: true },
                        },
                    }),
                };
            }
            return cap;
        });

        const result = await Runtime.answer(createInput({
            question: "Analisa com base em 20000 mensagens.",
        }));

        expect(result.answer).toBe("Agora com pesquisa.");
        expect(generateSpy).toHaveBeenCalledTimes(3);
    });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import path from "path";
import { SettingsService } from "@/app/SettingsService";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { ModelGateway, type ToolChatResult } from "@/ai/ModelGateway";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import type { TurnInput } from "@/runtime/contracts";

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

function makeBlankResult(): ToolChatResult {
    return {
        content: null,
        toolCalls: [],
        finishReason: "stop",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

describe("runtime stall correction", () => {
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
        // Isolate the operational store per file: otherwise these Runtime.answer
        // runs hit whatever sqlite path leaked from another test file's
        // settings (and, in CI sandboxes without network, fail on DB writes).
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath: path.join(
                    process.cwd(),
                    "storage",
                    "test-runtime-store",
                    `runtime-stall-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
    });

    it("rejects a stalling finish answer and allows the model to retry", async () => {
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Síntese de teste.");
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
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Sintese de teste.");
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeFinishResult("Vou dar uma olhada agora!"))
            .mockResolvedValueOnce(makeFinishResult("Te aviso quando terminar!"))
            .mockResolvedValueOnce(makeFinishResult("Não encontrei evidência sobre isso."));

        const result = await Runtime.answer(createInput({ question: "O que o João disse?" }));

        expect(result.answer).toBe("Não encontrei evidência sobre isso.");
        expect(generateSpy).toHaveBeenCalledTimes(3);
    });

    it("does not reject a non-stalling direct answer", async () => {
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Sintese de teste.");
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeFinishResult("Olá! Como posso ajudar?"));

        const result = await Runtime.answer(createInput({ question: "Oi" }));

        expect(result.answer).toBe("Olá! Como posso ajudar?");
        expect(generateSpy).toHaveBeenCalledTimes(1);
    });

    it("defers finish when the model mixes it with executable tool calls", async () => {
        const mixed: ToolChatResult = {
            content: null,
            toolCalls: [
                {
                    id: "finish-mixed",
                    type: "function",
                    function: { name: "finish", arguments: JSON.stringify({ answer: "premature" }) },
                },
                {
                    id: "math-mixed",
                    type: "function",
                    function: { name: "evaluate_math", arguments: JSON.stringify({ expression: "2+2" }) },
                },
            ],
            finishReason: "tool_calls",
            model: "test-model",
            durationMs: 10,
            usage: null,
        };
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Sintese de teste.");
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(mixed)
            .mockResolvedValueOnce(makeFinishResult("4"));

        const result = await Runtime.answer(createInput({ question: "2+2?" }));

        expect(result.answer).toBe("4");
        expect(result.toolRuns).toEqual([
            expect.objectContaining({ tool: "evaluate_math" }),
        ]);
        expect(generateSpy).toHaveBeenCalledTimes(2);
    });

    it("rejects raw tool-call markup and asks the model to retry properly", async () => {
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Sintese de teste.");
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

    it("rejects plain tool_call markup (arg_key style) as malformed output", async () => {
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Sintese de teste.");
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeMalformedMarkupResult([
                "<tool_call>tool_search",
                "<arg_key>query</arg_key>",
                "<arg_value>artifact card send</arg_value>",
                "</tool_call>",
            ].join("\n")))
            .mockResolvedValueOnce(makeFinishResult("Consegui continuar normalmente."));

        const result = await Runtime.answer(createInput({ question: "Continua a busca." }));

        expect(result.answer).toBe("Consegui continuar normalmente.");
        expect(generateSpy).toHaveBeenCalledTimes(2);
    });

    it("retries a blank model completion instead of ending the turn in silence", async () => {
        const recordSpy = vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath: path.join(
                    process.cwd(),
                    "storage",
                    "test-runtime-store",
                    `runtime-stall-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Sintese de teste.");
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
        
            .mockResolvedValueOnce(makeBlankResult())
            .mockResolvedValueOnce(makeFinishResult("Agora sim, uma resposta."));

        const result = await Runtime.answer(createInput({ question: "Responde alguma coisa." }));

        expect(result.answer).toBe("Agora sim, uma resposta.");
        expect(generateSpy).toHaveBeenCalledTimes(2);
        const persisted = recordSpy.mock.calls.at(-1)?.[0] as { traceEvents: Array<{ label: string; detail: string }> };
        expect(persisted.traceEvents.some((t) => t.label === "blank_response")).toBe(true);
    });

    it("nudges differently when the completion was truncated at the output limit", async () => {
        const recordSpy = vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath: path.join(
                    process.cwd(),
                    "storage",
                    "test-runtime-store",
                    `runtime-stall-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Sintese de teste.");
        const generateSpy = vi.spyOn(ModelGateway, "generateWithTools")
        
            .mockResolvedValueOnce({ content: null, toolCalls: [], finishReason: "length", model: "test-model", durationMs: 10, usage: null })
            .mockResolvedValueOnce(makeFinishResult("Versão enxuta enviada."));

        const result = await Runtime.answer(createInput({ question: "faz melhor, usa tudo!" }));

        expect(result.answer).toBe("Versão enxuta enviada.");
        expect(generateSpy).toHaveBeenCalledTimes(2);
        const persisted = recordSpy.mock.calls.at(-1)?.[0] as { traceEvents: Array<{ label: string; detail: string }> };
        const truncated = persisted.traceEvents.find((t) => t.label === "truncated_response");
        expect(truncated).toBeDefined();
        // The second call's messages must contain the targeted truncation nudge.
        const secondCallMessages = generateSpy.mock.calls[1][0] as Array<{ role: string; content: string }>;
        expect(secondCallMessages.some((m) => m.role === "system" && m.content.includes("output token limit"))).toBe(true);
    });
});

describe("runtime start_long_task interception", () => {
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
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath: path.join(
                    process.cwd(),
                    "storage",
                    "test-runtime-store",
                    `runtime-stall-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
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
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Sintese de teste.");
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

        const result = await Runtime.answer(createInput({ question: "Colete 2000 mensagens" }));

        expect(result.answer).toBe("Não consegui coletar as mensagens.");
        // start_long_task + 3 finish attempts = 4 calls
        expect(generateSpy).toHaveBeenCalledTimes(4);
    });

    it("prepares long-task context without introducing a call cap", async () => {
        const recordSpy = vi.spyOn(DiscordMemoryService, "recordRuntimeRun")
            .mockResolvedValue(undefined);
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeStartLongTaskResult({ reason: "no estimates given" }))
            .mockResolvedValueOnce(makeFinishResult("Done."));

        const result = await Runtime.answer(createInput({ question: "Big task" }));

        expect(result.answer).toBe("Done.");
        const persistedArgs = recordSpy.mock.calls.at(-1)?.[0] as { traceEvents: Array<{ label: string; detail: string }> };
        const longTaskTrace = persistedArgs.traceEvents.find((t) => t.label === "long_task");
        expect(longTaskTrace?.detail).toContain("Long-task context prepared");
        expect(longTaskTrace?.detail).not.toContain("calls=");
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
});

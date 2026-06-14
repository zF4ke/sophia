import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway, type ToolChatResult } from "@/ai/ModelGateway";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { Runtime } from "@/runtime/Runtime";
import type { TurnInput } from "@/runtime/contracts";

function createInput(overrides: Partial<TurnInput> = {}): TurnInput {
    return {
        question: "Faça uma análise das últimas 20000 mensagens do canal",
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

function makeRawTextResult(content: string): ToolChatResult {
    return {
        content,
        toolCalls: [],
        finishReason: "stop",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

function makeRetrievePayload(accumulated: number, continuationAvailable = true) {
    return {
        mode: "history",
        historyMessageCount: accumulated,
        accumulatedUniqueCount: accumulated,
        sourceOrigin: "history",
        continuation: {
            continuationAvailable,
            history: {
                continuationAvailable,
                perChannelOldestMessageId: continuationAvailable ? { c1: `oldest-${accumulated}` } : {},
            },
        },
        exhaustion: { historyExhausted: !continuationAvailable, exhaustedChannelIds: [] },
        messages: [],
    };
}

describe("runtime corpus task state machine", () => {
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
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({ stall: false });
    });

    it("rejects under-target finish, rejects raw-text, forces continuation, and falls back to deterministic incomplete string", async () => {
        // Mock retrieve_messages capability — each call returns the next page.
        let page = 0;
        const runMock = vi.fn().mockImplementation(async () => {
            page += 1;
            const accumulated = page * 1000;
            return {
                tool: "retrieve_messages",
                summary: `page ${page}: ${accumulated} messages so far`,
                data: makeRetrievePayload(accumulated, true),
            };
        });
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "retrieve_messages") return { ...cap, run: runMock };
            return cap;
        });

        // Model sequence:
        //   1) retrieve_messages(limit=20000) → page 1 (1000 collected)
        //   2) finish("early") → guard rejects (violation 1)
        //   3) raw-text answer → guard rejects (violation 2) → runtime forces continuation
        //   4) model stalls with no tool calls and no content → long_task_synthesis
        //      is skipped in favor of deterministic incomplete fallback.
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult("retrieve_messages", { query: "análise", limit: 20000, channelIds: ["c1"] }))
            .mockResolvedValueOnce(makeFinishResult("Resposta antecipada."))
            .mockResolvedValueOnce(makeRawTextResult("Baseado no que eu tenho, a pessoa é..."))
            .mockResolvedValue({ content: null, toolCalls: [], finishReason: "stop", model: "t", durationMs: 1, usage: null });

        const result = await Runtime.answer(createInput());

        // Unified guard + forced continuation + deterministic fallback:
        //   - guard rejected an under-target finish AND an under-target raw text
        //   - the runtime executed retrieve_messages itself after the second violation
        //   - the final answer is the deterministic bilingual incomplete string,
        //     NOT a fabricated analysis
        const recordToolRunSpy = DiscordMemoryService.recordToolRun as unknown as ReturnType<typeof vi.fn>;
        const retrieveRuns = recordToolRunSpy.mock.calls.filter((c) => c[5] === "retrieve_messages");

        // 1 model-driven + at least 1 runtime-forced continuation recorded via
        // the same persistence path as a normal tool run.
        expect(retrieveRuns.length).toBeGreaterThanOrEqual(2);
        expect(result.answer).toMatch(/Tarefa não concluída/);
        expect(result.answer).toMatch(/Task incomplete/);
        // The count shown in the fallback must come from the real retrieval state,
        // never the requested total.
        expect(result.answer).not.toMatch(/20000 de 20000/);
        expect(result.answer).toContain("20000"); // requested total is present
    });

    it("does not activate when the model pages with the default limit only", async () => {
        const runMock = vi.fn().mockResolvedValue({
            tool: "retrieve_messages",
            summary: "ok",
            data: makeRetrievePayload(500, false),
        });
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "retrieve_messages") return { ...cap, run: runMock };
            return cap;
        });

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult("retrieve_messages", { query: "q", limit: 1000, channelIds: ["c1"] }))
            .mockResolvedValueOnce(makeFinishResult("Pronto — peguei o que estava disponível."));

        const result = await Runtime.answer(createInput({ question: "Mostre algumas mensagens" }));

        // No corpus task activated → finish accepted normally, no forced synthesis fallback.
        expect(result.answer).toBe("Pronto — peguei o que estava disponível.");
    });

    it("does not activate from unrelated numbers like channel mentions or years", async () => {
        const runMock = vi.fn().mockResolvedValue({
            tool: "retrieve_messages",
            summary: "ok",
            data: makeRetrievePayload(100, true),
        });
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "retrieve_messages") return { ...cap, run: runMock };
            return cap;
        });

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult("retrieve_messages", { query: "q", limit: 100, channelIds: ["c1"] }))
            .mockResolvedValueOnce(makeFinishResult("Resposta normal."));

        const result = await Runtime.answer(createInput({
            question: "Analise o canal <#333333333333333333> em 2023 e me diga o clima geral",
        }));

        expect(result.answer).toBe("Resposta normal.");
    });

    it("activates from the user's requested corpus size even when the model starts with limit 100", async () => {
        let page = 0;
        const runMock = vi.fn().mockImplementation(async () => {
            page += 1;
            return {
                tool: "retrieve_messages",
                summary: `page ${page}`,
                data: makeRetrievePayload(page * 100, true),
            };
        });
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "retrieve_messages") return { ...cap, run: runMock };
            return cap;
        });

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult("retrieve_messages", { query: "análise", limit: 100, channelIds: ["c1"] }))
            .mockResolvedValueOnce(makeFinishResult("Resposta antecipada."))
            .mockResolvedValueOnce(makeRawTextResult("Ainda dá para responder com isso."))
            .mockResolvedValue({ content: null, toolCalls: [], finishReason: "stop", model: "t", durationMs: 1, usage: null });

        const result = await Runtime.answer(createInput({
            question: "Faça uma análise baseada nas últimas 20000 mensagens do canal",
        }));

        const recordToolRunSpy = DiscordMemoryService.recordToolRun as unknown as ReturnType<typeof vi.fn>;
        const retrieveRuns = recordToolRunSpy.mock.calls.filter((c) => c[5] === "retrieve_messages");
        expect(retrieveRuns.length).toBeGreaterThanOrEqual(2);
        expect(result.answer).toMatch(/Tarefa não concluída/);
        expect(result.answer).toMatch(/Task incomplete/);
    });

    it("allows finish once history is exhausted even if below requested total", async () => {
        const runMock = vi.fn().mockResolvedValue({
            tool: "retrieve_messages",
            summary: "exhausted at 500",
            data: makeRetrievePayload(500, false), // continuationAvailable=false, historyExhausted=true
        });
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        vi.spyOn(CapabilityRegistry, "get").mockImplementation((id) => {
            const cap = originalGet(id);
            if (id === "retrieve_messages") return { ...cap, run: runMock };
            return cap;
        });

        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult("retrieve_messages", { query: "q", limit: 20000, channelIds: ["c1"] }))
            .mockResolvedValueOnce(makeFinishResult("Só existem 500 mensagens no canal."));

        const result = await Runtime.answer(createInput());

        // Exhaustion lets the finish through — the guard is about truth, not quota.
        expect(result.answer).toBe("Só existem 500 mensagens no canal.");
    });
});

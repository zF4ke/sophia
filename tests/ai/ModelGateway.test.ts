import { ModelUsage } from "@/ai/ModelUsage";
import fs from "fs";
import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptyModelOutputError } from "@/ai/EmptyModelOutputError";
import { ModelGateway } from "@/ai/ModelGateway";
import * as modelProfiles from "@/app/modelProfiles";
import { AppPaths } from "@/app/AppPaths";
import { taskStore } from "@/runtime/tasks/TaskStore";

vi.mock("@/app/AppConfig", () => ({
    getAppConfig: () => ({
        discordToken: "discord-token",
        openRouterApiKey: "openrouter-key",
        openRouterBaseUrl: "https://openrouter.test/api/v1",
        modelProfileName: "test",
        modelProfile: {
            chatModel: "test-model",
            embeddingModel: "test-embedding",
            temperature: 0.2,
            maxOutputTokens: 256,
            contextWindow: 128000,
        },
        runtime: {
            operationalDbPath: "storage/runtime/operational.sqlite",
            toolCallLimit: 0,
            maxRepeatedCallSignature: 1,
            maxPriorTurns: 5,
            maxChannelMessages: 15,
            maxToolRunsContext: 12,
            maxEvidenceSlice: 32,
            escalationFetchLimit: 150,
            retrievalHistoryLimit: 50,
            retrievalContextWindow: 15,
        },
    }),
}));

function getLogFilePath(): string {
    const date = new Date().toISOString().slice(0, 10);
    return path.join(AppPaths.storageRoot, "logs", `model-output-${date}.jsonl`);
}

function readLogEntries(): Array<Record<string, unknown>> {
    const filePath = getLogFilePath();
    if (!fs.existsSync(filePath)) {
        return [];
    }

    return fs
        .readFileSync(filePath, "utf8")
        .trim()
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);
}

describe("ModelGateway", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        const filePath = getLogFilePath();
        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
        }
        (ModelGateway as any).client = {
            chat: {
                completions: {
                    create: vi.fn(),
                },
            },
            embeddings: {
                create: vi.fn(),
            },
        };
    });

    it("logs text generations to disk", async () => {
        const create = vi.fn().mockResolvedValue({
            choices: [{ message: { content: "Resposta final" } }],
            usage: {
                server_tool_use: {
                    web_search_requests: 0,
                },
            },
        });
        (ModelGateway as any).client.chat.completions.create = create;

        const output = await ModelGateway.generateText(
            [{ role: "user", content: "Pergunta de teste" }],
            {
                traceContext: {
                    traceLabel: "text_test",
                    questionPreview: "Pergunta de teste",
                },
            }
        );

        expect(output).toBe("Resposta final");
        const entries = readLogEntries();
        expect(entries).toHaveLength(1);
        expect(entries[0]).toMatchObject({
            callKind: "text",
            traceLabel: "text_test",
            model: "test-model",
            rawOutput: "Resposta final",
            normalizedOutput: "Resposta final",
            questionPreview: "Pergunta de teste",
            webMode: "off",
            webStatus: "off",
        });
    });

    it("records every retry attempt at the provider boundary", async () => {
        const record = vi.spyOn(taskStore, "recordModelUsage").mockResolvedValue(undefined);
        const previousBackoff = (ModelGateway as any).rateLimitBackoffMs;
        (ModelGateway as any).rateLimitBackoffMs = [0, 0];
        (ModelGateway as any).client.chat.completions.create = vi.fn()
            .mockRejectedValueOnce(Object.assign(new Error("throttled"), { status: 429 }))
            .mockResolvedValueOnce({ choices: [{ message: { content: "ready" } }], usage: { prompt_tokens: 10, completion_tokens: 2 } });
        try {
            expect(await ModelGateway.generateText([{ role: "user", content: "hello" }])).toBe("ready");
            expect(record.mock.calls.map(([row]) => [row.status, row.promptTokens])).toEqual([["failed", null], ["returned", 10]]);
        } finally { (ModelGateway as any).rateLimitBackoffMs = previousBackoff; }
    });

    it("logs json generations with parsed output", async () => {
        (ModelGateway as any).client.chat.completions.create = vi.fn().mockResolvedValue({
            choices: [{ message: { content: '{"mode":"direct_answer","reason":"ok"}' } }],
            usage: {
                server_tool_use: {
                    web_search_requests: 0,
                },
            },
        });

        const result = await ModelGateway.generateJson(
            [{ role: "user", content: "Classifique isso" }],
            { mode: "direct_answer", reason: "fallback" },
            {
                traceContext: {
                    traceLabel: "json_test",
                    questionPreview: "Classifique isso",
                },
            }
        );

        expect(result).toEqual({ mode: "direct_answer", reason: "ok" });
        const entries = readLogEntries();
        expect(entries).toHaveLength(1);
        expect(entries[0]).toMatchObject({
            callKind: "json",
            traceLabel: "json_test",
            parsedJson: { mode: "direct_answer", reason: "ok" },
        });
    });

    it("logs router json generations with a distinct trace label", async () => {
        (ModelGateway as any).client.chat.completions.create = vi.fn().mockResolvedValue({
            choices: [
                {
                    message: {
                        content: '{"intent":"channel_target","targetText":"scart","confidence":0.91,"reason":"channel"}',
                    },
                },
            ],
            usage: {
                server_tool_use: {
                    web_search_requests: 0,
                },
            },
        });

        await ModelGateway.generateJson(
            [{ role: "user", content: "Route this" }],
            { intent: "broad_search", reason: "fallback" },
            {
                traceContext: {
                    traceLabel: "discord_question_routing",
                    questionPreview: "Route this",
                },
            }
        );

        const entries = readLogEntries();
        expect(entries).toHaveLength(1);
        expect(entries[0]).toMatchObject({
            callKind: "json",
            traceLabel: "discord_question_routing",
        });
    });

    it("throws on blank text output and logs the failure", async () => {
        (ModelGateway as any).client.chat.completions.create = vi.fn().mockResolvedValue({
            choices: [{ message: { content: "   " } }],
            usage: {
                server_tool_use: {
                    web_search_requests: 0,
                },
            },
        });

        await expect(
            ModelGateway.generateText([{ role: "user", content: "Pergunta vazia" }], {
                traceContext: {
                    traceLabel: "blank_text_test",
                },
            })
        ).rejects.toBeInstanceOf(EmptyModelOutputError);

        const entries = readLogEntries();
        expect(entries).toHaveLength(1);
        expect(entries[0]).toMatchObject({
            callKind: "text",
            traceLabel: "blank_text_test",
            blankOutput: true,
        });
    });

    it("throws when text generation emits raw tool markup", async () => {
        (ModelGateway as any).client.chat.completions.create = vi.fn().mockResolvedValue({
            choices: [{
                message: {
                    content: [
                        "<minimax:tool_call>",
                        "<invoke name=\"resolve_member_identity\">",
                        "<parameter name=\"targets\">[\"riverside\"]</parameter>",
                        "</invoke>",
                        "</minimax:tool_call>",
                    ].join("\n"),
                },
            }],
            usage: {
                server_tool_use: {
                    web_search_requests: 0,
                },
            },
        });

        await expect(
            ModelGateway.generateText([{ role: "user", content: "Continue" }], {
                traceContext: {
                    traceLabel: "raw_tool_markup_text",
                },
            })
        ).rejects.toThrow("Model emitted raw tool markup during text generation");
    });

    it("sends the OpenRouter web search tool when web mode is enabled", async () => {
        const create = vi.fn().mockResolvedValue({
            choices: [{ message: { content: "Resposta com web" } }],
            usage: {
                server_tool_use: {
                    web_search_requests: 2,
                },
            },
        });
        (ModelGateway as any).client.chat.completions.create = create;

        await ModelGateway.generateText(
            [{ role: "user", content: "Quais foram as noticias de IA hoje?" }],
            {
                webMode: "auto",
                traceContext: {
                    traceLabel: "web_text_test",
                    questionPreview: "Quais foram as noticias de IA hoje?",
                    webMode: "auto",
                    webContext: "talk_direct_answer",
                },
            }
        );

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                provider: { sort: "throughput" },
                tools: [
                    {
                        type: "openrouter:web_search",
                        parameters: {
                            engine: "auto",
                            max_results: 5,
                            search_context_size: "medium",
                        },
                    },
                ],
            }),
            { signal: undefined },
        );

        const entries = readLogEntries();
        expect(entries).toHaveLength(1);
        expect(entries[0]).toMatchObject({
            traceLabel: "web_text_test",
            webMode: "auto",
            webContext: "talk_direct_answer",
            webStatus: "used",
            webSearchRequests: 2,
        });
    });

    it("routes tool calls through the highest-throughput provider", async () => {
        const create = vi.fn().mockResolvedValue({
            choices: [{ message: { content: null, tool_calls: [] }, finish_reason: "stop" }],
        });
        (ModelGateway as any).client.chat.completions.create = create;

        await ModelGateway.generateWithTools(
            [{ role: "user", content: "Teste" }],
            { tools: [] }
        );

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                provider: { sort: "throughput" },
                parallel_tool_calls: false,
            }),
            { signal: undefined },
        );
    });

    it("sends local-endpoint requests without the OpenRouter routing block", async () => {
        const create = vi.fn().mockResolvedValue({
            choices: [{ message: { content: "resposta local" } }],
        });
        const localBase = "http://127.0.0.1:1234/v1";
        (ModelGateway as any).profileClients = new Map([
            [JSON.stringify([localBase, "not-needed"]), { chat: { completions: { create } } }],
        ]);

        const result = await ModelGateway.generateWithTools(
            [{ role: "user", content: "Teste" }],
            {
                tools: [],
                profile: {
                    chatModel: "local-model",
                    embeddingModel: "openai/text-embedding-3-small",
                    temperature: 0.5,
                    maxOutputTokens: 1024,
                    contextWindow: 32768,
                    baseUrl: localBase,
                },
                traceContext: { traceLabel: "local_test" },
            }
        );

        expect(create).toHaveBeenCalledTimes(1);
        const request = create.mock.calls[0][0] as Record<string, unknown>;
        expect(request.model).toBe("local-model");
        expect("provider" in request).toBe(false);
        expect(result.model).toBe("local-model");
    });

    it("retries once when the provider returns a malformed completion for tool chat", async () => {
        const create = vi
            .fn()
            .mockResolvedValueOnce({ id: "bad-response" })
            .mockResolvedValueOnce({
                choices: [
                    {
                        message: {
                            content: null,
                            tool_calls: [
                                {
                                    id: "tool_1",
                                    function: {
                                        name: "finish",
                                        arguments: "{\"answer\":\"ok\"}",
                                    },
                                },
                            ],
                        },
                        finish_reason: "tool_calls",
                    },
                ],
                usage: {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
            });
        (ModelGateway as any).client.chat.completions.create = create;

        const result = await ModelGateway.generateWithTools(
            [{ role: "user", content: "Teste" }],
            {
                tools: [],
                traceContext: {
                    traceLabel: "tool_retry_test",
                },
            }
        );

        expect(create).toHaveBeenCalledTimes(2);
        expect(result.toolCalls).toEqual([
            {
                id: "tool_1",
                type: "function",
                function: {
                    name: "finish",
                    arguments: "{\"answer\":\"ok\"}",
                },
            },
        ]);
    });

    it("flags raw invoke markup when the provider omits tool_calls", async () => {
        (ModelGateway as any).client.chat.completions.create = vi.fn().mockResolvedValue({
            choices: [
                {
                    message: {
                        content: [
                            "<invoke name=\"search__messages\">",
                            "<parameter name=\"author_id\">123</parameter>",
                            "<parameter name=\"limit\">25</parameter>",
                            "</invoke>",
                            "</minimax:tool_call>",
                        ].join("\n"),
                    },
                    finish_reason: "stop",
                },
            ],
            usage: {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        });

        const result = await ModelGateway.generateWithTools(
            [{ role: "user", content: "Teste" }],
            {
                tools: [],
                traceContext: {
                    traceLabel: "invoke_markup_test",
                },
            }
        );

        expect(result.content).toBeNull();
        expect(result.toolCalls).toEqual([]);
        expect(result.malformedToolCallText).toContain("<invoke name=\"search__messages\">");
    });

    it("retries provider rate limits with backoff and succeeds", async () => {
        (ModelGateway as any).rateLimitBackoffMs = [1, 1];
        const rateLimitError = Object.assign(new Error("Provider returned error"), { status: 429 });
        const create = vi.fn()
            .mockRejectedValueOnce(rateLimitError)
            .mockResolvedValueOnce({
                choices: [{ message: { content: "Sobrevivi ao 429" } }],
            });
        (ModelGateway as any).client.chat.completions.create = create;

        const result = await ModelGateway.generateWithTools(
            [{ role: "user", content: "Teste" }],
            { tools: [], traceContext: { traceLabel: "retry_test" } }
        );

        expect(create).toHaveBeenCalledTimes(2);
        expect(result.content).toBe("Sobrevivi ao 429");
    });

    it("does not retry non-retryable provider errors", async () => {
        (ModelGateway as any).rateLimitBackoffMs = [1, 1];
        const create = vi.fn()
            .mockRejectedValue(Object.assign(new Error("bad request"), { status: 400 }));
        (ModelGateway as any).client.chat.completions.create = create;

        await expect(
            ModelGateway.generateWithTools([{ role: "user", content: "Teste" }], { tools: [] })
        ).rejects.toThrow("bad request");
        expect(create).toHaveBeenCalledTimes(1);
    });

    it("does not send an external default model to OpenRouter after a failure", async () => {
        const config = modelProfiles.readModelProfiles();
        vi.spyOn(modelProfiles, "readModelProfiles").mockReturnValue({ ...config, defaultProfile: "external", profiles: { ...config.profiles, external: { ...config.profiles.glm53flash, chatModel: "external-model", baseUrl: "https://example.com/v1", api: "responses" } } });
        (ModelGateway as any).rateLimitBackoffMs = [1, 1];
        const create = vi.fn().mockRejectedValue(Object.assign(new Error("rate limited"), { status: 429 }));
        (ModelGateway as any).client.chat.completions.create = create;
        await expect(ModelGateway.generateWithTools([{ role: "user", content: "Test" }], { tools: [] })).rejects.toThrow("rate limited");
        expect(create).toHaveBeenCalledTimes(3);
        expect(create.mock.calls.some(([request]) => request.model === "external-model")).toBe(false);
    });
    it("falls back to the default model when a secondary model keeps rate-limiting", async () => {
        const config = modelProfiles.readModelProfiles();
        vi.spyOn(modelProfiles, "readModelProfiles").mockReturnValue({ ...config, defaultProfile: "glm53flash" });
        (ModelGateway as any).rateLimitBackoffMs = [1, 1];
        const rateLimitError = Object.assign(new Error("Provider returned error"), { status: 429 });
        const create = vi.fn().mockRejectedValue(rateLimitError);
        // Succeed only when the request switched to the default profile's model.
        create.mockImplementation(async (request: { model: string }) => {
            if (request.model === "z-ai/glm-5.3-flash") {
                return { choices: [{ message: { content: "fallback!" } }] };
            }
            throw rateLimitError;
        });
        (ModelGateway as any).client.chat.completions.create = create;

        const result = await ModelGateway.generateWithTools(
            [{ role: "user", content: "Teste" }],
            { tools: [], traceContext: { traceLabel: "fallback_test" } }
        );

        expect(create.mock.calls.some(([request]) => (request as { model: string }).model === "z-ai/glm-5.3-flash")).toBe(true);
        expect(result.model).toBe("z-ai/glm-5.3-flash");
        expect(result.content).toBe("fallback!");
    });
});

it("passes execution cancellation to active chat-completions requests", async () => {
    const controller = new AbortController();
    const create = vi.fn((_body, options) => new Promise((_resolve, reject) => {
        expect(options.signal).toBe(controller.signal);
        options.signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
        controller.abort();
    }));
    await expect(ModelUsage.scope({ actorId: "cancel-test" }, () => {
        ModelUsage.bindExecution(controller.signal);
        return (ModelGateway as any).requestWithProviderRetries({ chat: { completions: { create } } }, {}, { chatModel: "test" }, "test");
    })).rejects.toThrow("cancelled");
    expect(create).toHaveBeenCalledOnce();
});

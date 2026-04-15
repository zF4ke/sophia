import fs from "fs";
import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptyModelOutputError } from "@/ai/EmptyModelOutputError";
import { ModelGateway } from "@/ai/ModelGateway";
import { AppPaths } from "@/app/AppPaths";

vi.mock("@/app/AppConfig", () => ({
    getAppConfig: () => ({
        discordToken: "discord-token",
        openRouterApiKey: "openrouter-key",
        openRouterBaseUrl: "https://openrouter.test/api/v1",
        modelProfileName: "test",
        modelProfile: {
            chatModel: "test-model",
            analysisModel: "test-analysis",
            embeddingModel: "test-embedding",
            temperature: 0.2,
            maxOutputTokens: 256,
            contextWindow: 128000,
        },
        runtime: {
            operationalDbPath: "storage/runtime/operational.sqlite",
            checkpointDbPath: "storage/runtime/checkpoints.sqlite",
            maxToolCalls: 6,
            maxRepeatedCallSignature: 1,
            maxLatencyBudgetMs: 15000,
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
            })
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
});



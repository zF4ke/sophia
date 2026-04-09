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
        port: 3000,
        modelProfileName: "test",
        modelProfile: {
            chatModel: "test-model",
            analysisModel: "test-analysis",
            embeddingModel: "test-embedding",
            temperature: 0.2,
            maxOutputTokens: 256,
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
        });
    });

    it("logs json generations with parsed output", async () => {
        (ModelGateway as any).client.chat.completions.create = vi.fn().mockResolvedValue({
            choices: [{ message: { content: '{"mode":"direct_answer","reason":"ok"}' } }],
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

    it("throws on blank text output and logs the failure", async () => {
        (ModelGateway as any).client.chat.completions.create = vi.fn().mockResolvedValue({
            choices: [{ message: { content: "   " } }],
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
});

import OpenAI from "openai";
import { EmptyModelOutputError } from "@/ai/EmptyModelOutputError";
import { ModelTraceLogger } from "@/ai/ModelTraceLogger";
import { getAppConfig } from "@/app/AppConfig";
import type { ModelProfile, ModelTraceContext } from "@/shared/appTypes";

type ChatMessage = {
    role: "system" | "user" | "assistant";
    content: string;
};

type ChatOptions = {
    profile?: ModelProfile;
    temperature?: number;
    maxOutputTokens?: number;
    traceContext?: ModelTraceContext;
};

export class ModelGateway {
    private static client: OpenAI | null = null;

    private static getClient(): OpenAI {
        if (!this.client) {
            const config = getAppConfig();
            this.client = new OpenAI({
                apiKey: config.openRouterApiKey,
                baseURL: config.openRouterBaseUrl,
                defaultHeaders: {
                    "HTTP-Referer": "https://sophia3.local",
                    "X-OpenRouter-Title": "Sophia3",
                },
            });
        }

        return this.client;
    }

    public static async generateText(
        messages: ChatMessage[],
        options: ChatOptions = {}
    ): Promise<string> {
        const { model, rawOutput, durationMs } = await this.runChatCompletion(messages, options);
        const normalizedOutput = rawOutput.trim();
        const traceLabel = options.traceContext?.traceLabel || "unlabeled_text_generation";

        ModelTraceLogger.log({
            timestamp: new Date().toISOString(),
            callKind: "text",
            model,
            traceLabel,
            questionPreview: this.getQuestionPreview(messages, options.traceContext),
            durationMs,
            messages,
            rawOutput,
            normalizedOutput,
            blankOutput: !normalizedOutput,
        });

        if (!normalizedOutput) {
            throw new EmptyModelOutputError(traceLabel);
        }

        return normalizedOutput;
    }

    public static async generateJson<T>(
        messages: ChatMessage[],
        fallback: T,
        options: ChatOptions = {}
    ): Promise<T> {
        const { model, rawOutput, durationMs } = await this.runChatCompletion(messages, options);
        const traceLabel = options.traceContext?.traceLabel || "unlabeled_json_generation";
        const extracted = this.extractJson(rawOutput);
        let parsedJson: T | undefined;
        let parseError: string | undefined;

        if (!rawOutput.trim()) {
            parseError = "Model returned empty output.";
        } else if (!extracted) {
            parseError = "No JSON object found in model output.";
        } else {
            try {
                parsedJson = JSON.parse(extracted) as T;
            } catch (error) {
                parseError =
                    error instanceof Error ? error.message : "Failed to parse JSON output.";
            }
        }

        ModelTraceLogger.log({
            timestamp: new Date().toISOString(),
            callKind: "json",
            model,
            traceLabel,
            questionPreview: this.getQuestionPreview(messages, options.traceContext),
            durationMs,
            messages,
            rawOutput,
            normalizedOutput: rawOutput.trim(),
            blankOutput: !rawOutput.trim(),
            parsedJson,
            parseError,
        });

        if (parsedJson !== undefined) {
            return parsedJson;
        }

        return fallback;
    }

    public static async embedTexts(texts: string[]): Promise<number[][]> {
        if (!texts.length) {
            return [];
        }

        const config = getAppConfig();
        const client = this.getClient();
        const response = await client.embeddings.create({
            model: config.modelProfile.embeddingModel,
            input: texts,
        });

        return response.data.map((item) => item.embedding as number[]);
    }

    private static extractJson(raw: string): string | null {
        const trimmed = raw.trim();
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            return trimmed;
        }

        const match = trimmed.match(/\{[\s\S]*\}/);
        return match ? match[0] : null;
    }

    private static async runChatCompletion(
        messages: ChatMessage[],
        options: ChatOptions
    ): Promise<{ model: string; rawOutput: string; durationMs: number }> {
        const config = getAppConfig();
        const profile = options.profile || config.modelProfile;
        const client = this.getClient();
        const startedAt = Date.now();
        const completion = await client.chat.completions.create({
            model: profile.chatModel,
            temperature: options.temperature ?? profile.temperature,
            max_tokens: options.maxOutputTokens ?? profile.maxOutputTokens,
            messages,
        });

        return {
            model: profile.chatModel,
            rawOutput: completion.choices[0]?.message?.content || "",
            durationMs: Math.max(0, Date.now() - startedAt),
        };
    }

    private static getQuestionPreview(
        messages: ChatMessage[],
        traceContext?: ModelTraceContext
    ): string | null {
        if (traceContext?.questionPreview?.trim()) {
            return traceContext.questionPreview.trim().slice(0, 200);
        }

        const lastUserMessage = [...messages]
            .reverse()
            .find((message) => message.role === "user" && message.content.trim());
        return lastUserMessage ? lastUserMessage.content.trim().slice(0, 200) : null;
    }
}

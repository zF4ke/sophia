import OpenAI from "openai";
import { EmptyModelOutputError } from "@/ai/EmptyModelOutputError";
import { ModelTraceLogger } from "@/ai/ModelTraceLogger";
import { getAppConfig } from "@/app/AppConfig";
import type {
    ModelProfile,
    ModelTraceContext,
    WebMode,
    WebStatus,
} from "@/shared/appTypes";

type ChatMessage = {
    role: "system" | "user" | "assistant";
    content: string;
};

export type ToolCallMessage = {
    role: "assistant";
    content: string | null;
    tool_calls: ToolCall[];
};

export type ToolResultMessage = {
    role: "tool";
    tool_call_id: string;
    content: string;
};

export type ToolCall = {
    id: string;
    type: "function";
    function: {
        name: string;
        arguments: string;
    };
};

export type ToolChatMessage = ChatMessage | ToolCallMessage | ToolResultMessage;

type NativeToolDef = {
    type: "function";
    function: {
        name: string;
        description: string;
        parameters: Record<string, unknown>;
    };
};

type ChatOptions = {
    profile?: ModelProfile;
    temperature?: number;
    maxOutputTokens?: number;
    traceContext?: ModelTraceContext;
    webMode?: WebMode;
    onComplete?: (meta: {
        webStatus: WebStatus;
        webSearchRequests: number;
    }) => void | Promise<void>;
};

export type ToolChatOptions = Omit<ChatOptions, "webMode"> & {
    tools: NativeToolDef[];
};

export type TokenUsage = {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
};

export type ToolChatResult = {
    content: string | null;
    toolCalls: ToolCall[];
    malformedToolCallText?: string | null;
    finishReason: string;
    model: string;
    durationMs: number;
    usage: TokenUsage | null;
};

export class ModelGateway {
    private static client: OpenAI | null = null;
    private static readonly MALFORMED_COMPLETION_RETRIES = 1;

    private static getClient(): OpenAI {
        if (!this.client) {
            const config = getAppConfig();
            this.client = new OpenAI({
                apiKey: config.openRouterApiKey,
                baseURL: config.openRouterBaseUrl,
                defaultHeaders: {
                    "HTTP-Referer": "https://sophia.local",
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
        const { model, rawOutput, durationMs, webStatus, webSearchRequests } =
            await this.runChatCompletion(messages, options);
        const normalizedOutput = rawOutput.trim();
        const traceLabel = options.traceContext?.traceLabel || "unlabeled_text_generation";

        ModelTraceLogger.log({
            timestamp: new Date().toISOString(),
            callKind: "text",
            model,
            traceLabel,
            questionPreview: this.getQuestionPreview(messages, options.traceContext),
            durationMs,
            webMode: options.webMode || "off",
            webContext: options.traceContext?.webContext,
            webStatus,
            webSearchRequests,
            messages,
            rawOutput,
            normalizedOutput,
            blankOutput: !normalizedOutput,
            traceEvents: options.traceContext?.traceEvents,
        });

        await options.onComplete?.({
            webStatus,
            webSearchRequests,
        });

        if (normalizedOutput && this.containsInvokeMarkup(normalizedOutput)) {
            throw new Error(`Model emitted raw tool markup during text generation (${traceLabel}).`);
        }

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
        const { model, rawOutput, durationMs, webStatus, webSearchRequests } =
            await this.runChatCompletion(messages, options);
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
            webMode: options.webMode || "off",
            webContext: options.traceContext?.webContext,
            webStatus,
            webSearchRequests,
            messages,
            rawOutput,
            normalizedOutput: rawOutput.trim(),
            blankOutput: !rawOutput.trim(),
            parsedJson,
            parseError,
        });

        await options.onComplete?.({
            webStatus,
            webSearchRequests,
        });

        if (parsedJson !== undefined) {
            return parsedJson;
        }

        return fallback;
    }

    public static async generateWithTools(
        messages: ToolChatMessage[],
        options: ToolChatOptions
    ): Promise<ToolChatResult> {
        const config = getAppConfig();
        const profile = options.profile || config.modelProfile;
        const client = this.getClient();
        const startedAt = Date.now();
        const traceLabel = options.traceContext?.traceLabel || "unlabeled_tool_chat";

        const request: Record<string, unknown> = {
            model: profile.chatModel,
            temperature: options.temperature ?? profile.temperature,
            max_tokens: options.maxOutputTokens ?? profile.maxOutputTokens,
            messages,
            tools: options.tools,
        };

        const completion = await this.createChatCompletionWithValidation(
            client,
            request,
            traceLabel,
            profile.chatModel
        );
        const durationMs = Math.max(0, Date.now() - startedAt);
        const choice = completion.choices[0];
        const rawContent = this.normalizeAssistantContent(choice?.message?.content);
        const structuredToolCalls: ToolCall[] = (choice?.message?.tool_calls ?? []).map((tc: any) => ({
            id: tc.id,
            type: "function" as const,
            function: {
                name: tc.function.name,
                arguments: tc.function.arguments,
            },
        }));
        const malformedToolCallText =
            structuredToolCalls.length === 0 && rawContent && this.containsInvokeMarkup(rawContent)
                ? rawContent
                : null;
        const toolCalls = structuredToolCalls;
        const content = malformedToolCallText ? null : rawContent;
        const finishReason = choice?.finish_reason || "stop";

        ModelTraceLogger.log({
            timestamp: new Date().toISOString(),
            callKind: "tool_chat",
            model: profile.chatModel,
            traceLabel,
            questionPreview: this.getQuestionPreview(
                messages.filter((m): m is ChatMessage => m.role !== "tool") as ChatMessage[],
                options.traceContext
            ),
            durationMs,
            webMode: "off",
            webStatus: "off",
            webSearchRequests: 0,
            messages,
            rawOutput: content || "",
            normalizedOutput: content?.trim() || "",
            blankOutput: !content?.trim() && toolCalls.length === 0,
            toolCalls,
            finishReason,
            traceEvents: options.traceContext?.traceEvents,
        });

        const usage: TokenUsage | null = completion.usage
            ? {
                promptTokens: completion.usage.prompt_tokens ?? 0,
                completionTokens: completion.usage.completion_tokens ?? 0,
                totalTokens: completion.usage.total_tokens ?? 0,
            }
            : null;

        return { content, toolCalls, malformedToolCallText, finishReason, model: profile.chatModel, durationMs, usage };
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
    ): Promise<{
        model: string;
        rawOutput: string;
        durationMs: number;
        webStatus: WebStatus;
        webSearchRequests: number;
    }> {
        const config = getAppConfig();
        const profile = options.profile || config.modelProfile;
        const client = this.getClient();
        const startedAt = Date.now();
        const webMode = options.webMode || "off";
        const request: Record<string, unknown> = {
            model: profile.chatModel,
            temperature: options.temperature ?? profile.temperature,
            max_tokens: options.maxOutputTokens ?? profile.maxOutputTokens,
            messages,
        };

        if (webMode !== "off") {
            request.tools = [
                {
                    type: "openrouter:web_search",
                    parameters: {
                        engine: "auto",
                        max_results: 5,
                        search_context_size: "medium",
                    },
                },
            ];
        }

        const completion = await this.createChatCompletionWithValidation(
            client,
            request,
            options.traceContext?.traceLabel || "unlabeled_text_generation",
            profile.chatModel
        );
        const webSearchRequests = Math.max(
            0,
            Number((completion as any)?.usage?.server_tool_use?.web_search_requests ?? 0)
        );
        const webStatus: WebStatus =
            webMode === "off" ? "off" : webSearchRequests > 0 ? "used" : "enabled";

        return {
            model: profile.chatModel,
            rawOutput: completion.choices[0]?.message?.content || "",
            durationMs: Math.max(0, Date.now() - startedAt),
            webStatus,
            webSearchRequests,
        };
    }

    private static async createChatCompletionWithValidation(
        client: OpenAI,
        request: Record<string, unknown>,
        traceLabel: string,
        model: string
    ) {
        let lastMalformedPayload: unknown = null;

        for (let attempt = 0; attempt <= this.MALFORMED_COMPLETION_RETRIES; attempt += 1) {
            const completion = await client.chat.completions.create(request as any);
            if (Array.isArray((completion as any)?.choices) && (completion as any).choices.length > 0) {
                return completion;
            }

            lastMalformedPayload = completion;
        }

        throw new Error(
            `Model provider returned no choices for ${model} (${traceLabel}). Payload=${this.safeSerialize(lastMalformedPayload)}`
        );
    }

    private static safeSerialize(value: unknown): string {
        try {
            return JSON.stringify(value);
        } catch {
            return String(value);
        }
    }

    private static containsInvokeMarkup(value: string): boolean {
        return /<invoke\s+name="[^"]+"\s*>/i.test(value) && /<\/invoke>/i.test(value);
    }

    private static normalizeAssistantContent(content: unknown): string | null {
        if (typeof content === "string") {
            return content;
        }
        if (Array.isArray(content)) {
            const text = content
                .map((item) => {
                    if (typeof item === "string") return item;
                    if (item && typeof item === "object" && "text" in item && typeof (item as any).text === "string") {
                        return (item as any).text;
                    }
                    return "";
                })
                .join("\n")
                .trim();
            return text || null;
        }
        return null;
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

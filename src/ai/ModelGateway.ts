import OpenAI from "openai";
import { ModelUsage } from "./ModelUsage";
import { EmptyModelOutputError } from "@/ai/EmptyModelOutputError";
import { ModelTraceLogger } from "@/ai/ModelTraceLogger";
import { getAppConfig } from "@/app/AppConfig";
import { profileUsesResponsesApi, callResponsesApi } from "@/ai/ResponsesAdapter";
import { readModelProfiles } from "@/app/modelProfiles";
import type {
    ModelProfile,
    ModelTraceContext,
    WebMode,
    WebStatus,
} from "@/shared/appTypes";

type ChatMessage = {
    role: "system" | "user" | "assistant";
    content: string;
    images?: Array<{ url: string; detail?: "auto" | "low" | "high" }>;
    audio?: Array<{ data: string; format: "wav" | "mp3" }>;
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

export function providerMessages(messages: ToolChatMessage[]): unknown[] {
    return messages.map(message => {
        if (message.role !== "user") return message;
        const { images, audio, ...textMessage } = message;
        if (!images?.length && !audio?.length) return textMessage;
        return { ...textMessage, content: [{ type: "text", text: message.content },
            ...(images ?? []).map(image => ({ type: "image_url", image_url: { url: image.url, detail: image.detail ?? "auto" } })),
            ...(audio ?? []).map(clip => ({ type: "input_audio", input_audio: clip }))] };
    });
}

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
    /** Extra clients keyed by base URL, for per-profile endpoints (LM Studio, etc.). */
    private static readonly profileClients = new Map<string, OpenAI>();
    private static readonly MALFORMED_COMPLETION_RETRIES = 1;
    private static readonly RATE_LIMIT_RETRIES = 2;
    /** Per-request HTTP timeout. Deliberately generous: it bounds a single hung
     * connection, not the model's total work (turns have no wall-clock cap). */
    private static readonly REQUEST_TIMEOUT_MS = 600_000;
    /** Backoff between provider-error retries. Mutable for tests. */
    private static rateLimitBackoffMs = [1_500, 4_000];

    private static getClient(): OpenAI {
        if (!this.client) {
            const config = getAppConfig();
            this.client = new OpenAI({
                apiKey: config.openRouterApiKey,
                baseURL: config.openRouterBaseUrl,
                timeout: this.REQUEST_TIMEOUT_MS,
                maxRetries: 0,
                defaultHeaders: {
                    "HTTP-Referer": "https://sophia.local",
                    "X-OpenRouter-Title": "Sophia",
                },
            });
        }

        return this.client;
    }

    /**
     * The client for a chat profile: OpenRouter by default, or the profile's
     * own OpenAI-compatible endpoint (OpenCode Zen, LM Studio, etc.).
     */
    private static async getClientForProfile(profile: ModelProfile): Promise<OpenAI> {
        const config = getAppConfig();
        if (!profile.baseUrl || profile.baseUrl === config.openRouterBaseUrl) {
            return this.getClient();
        }
        const apiKey = profile.apiKeyEnv ? process.env[profile.apiKeyEnv] : "not-needed";
        if (!apiKey) throw new Error(`Missing ${profile.apiKeyEnv} for model profile ${profile.label ?? profile.chatModel}.`);
        const clientKey = JSON.stringify([profile.baseUrl, apiKey]);
        const cached = this.profileClients.get(clientKey);
        if (cached) return cached;
        const client = new OpenAI({
            apiKey,
            baseURL: profile.baseUrl,
            timeout: this.REQUEST_TIMEOUT_MS,
            maxRetries: 0,
        });
        this.profileClients.set(clientKey, client);
        return client;
    }

    /** True when the profile routes through OpenRouter (provider routing applies). */
    private static usesOpenRouter(profile: ModelProfile): boolean {
        const config = getAppConfig();
        return !profile.baseUrl || profile.baseUrl === config.openRouterBaseUrl;
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
        const client = await this.getClientForProfile(profile);
        const startedAt = Date.now();
        const traceLabel = options.traceContext?.traceLabel || "unlabeled_tool_chat";

        // Some providers (OpenCode Zen muse free models) only expose the
        // Responses API; /chat/completions 500s on them.
        if (!this.usesOpenRouter(profile) && profileUsesResponsesApi(profile)) {
            const startedAt2 = startedAt;
            const maxOutputTokens = options.maxOutputTokens ?? profile.maxOutputTokens;
            let outcome;
            try {
                outcome = await ModelUsage.measure(profile, traceLabel, () => callResponsesApi(client, profile, messages, {
                    tools: options.tools,
                    maxOutputTokens,
                    temperature: options.temperature ?? profile.temperature,
                }));
            } catch (error) {
                const durationMs = Math.max(0, Date.now() - startedAt2);
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
                    rawOutput: "",
                    normalizedOutput: "",
                    blankOutput: true,
                    toolCalls: [],
                    finishReason: "error",
                    traceEvents: options.traceContext?.traceEvents,
                });
                throw error;
            }
            const durationMs = Math.max(0, Date.now() - startedAt2);
            const malformedToolCallText =
                outcome.toolCalls.length === 0 && outcome.content && this.containsInvokeMarkup(outcome.content)
                    ? outcome.content
                    : null;
            const content = malformedToolCallText ? null : outcome.content;

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
                blankOutput: !content?.trim() && outcome.toolCalls.length === 0,
                toolCalls: outcome.toolCalls,
                finishReason: outcome.finishReason,
                traceEvents: options.traceContext?.traceEvents,
            });

            return {
                content,
                toolCalls: outcome.toolCalls,
                malformedToolCallText,
                finishReason: outcome.finishReason,
                model: profile.chatModel,
                durationMs,
                usage: outcome.usage,
            };
        }

        const request: Record<string, unknown> = {
            model: profile.chatModel,
            temperature: options.temperature ?? profile.temperature,
            max_tokens: options.maxOutputTokens ?? profile.maxOutputTokens,
            messages: providerMessages(messages),
            tools: options.tools,
            parallel_tool_calls: profile.parallelToolCalls ?? false,
        };
        // Provider routing is an OpenRouter-only request field.
        if (this.usesOpenRouter(profile)) {
            request.provider = { sort: "throughput" };
        }

        let modelUsed = profile.chatModel;
        const completion = await this.createChatCompletionWithValidation(
            async () => {
                const outcome = await this.createWithModelFallback(client, request, profile, traceLabel);
                modelUsed = outcome.modelUsed;
                return outcome.completion;
            },
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
            model: modelUsed,
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

        return { content, toolCalls, malformedToolCallText, finishReason, model: modelUsed, durationMs, usage };
    }

    public static async embedTexts(texts: string[]): Promise<number[][]> {
        if (!texts.length) {
            return [];
        }

        const config = getAppConfig();
        // Embeddings always route through OpenRouter, even when the chat
        // profile points at a local server: local embedding models are not
        // part of the retrieval contract.
        const client = this.getClient();
        const response = await ModelUsage.measure({ chatModel: config.modelProfile.embeddingModel }, "embeddings", () => client.embeddings.create({
            model: config.modelProfile.embeddingModel,
            input: texts,
        }));

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
        const client = await this.getClientForProfile(profile);
        const startedAt = Date.now();
        const webMode = options.webMode || "off";

        // Responses-API profiles (OpenCode Zen muse free models): plain text
        // generation must also go through /responses or it 500s.
        if (!this.usesOpenRouter(profile) && profileUsesResponsesApi(profile)) {
            const outcome = await ModelUsage.measure(profile, options.traceContext?.traceLabel ?? "text", () => callResponsesApi(client, profile, messages, {
                tools: [],
                maxOutputTokens: options.maxOutputTokens ?? profile.maxOutputTokens,
                temperature: options.temperature ?? profile.temperature,
            }));
            return {
                model: profile.chatModel,
                rawOutput: outcome.content || "",
                durationMs: Math.max(0, Date.now() - startedAt),
                webStatus: "off" as WebStatus,
                webSearchRequests: 0,
            };
        }

        const request: Record<string, unknown> = {
            model: profile.chatModel,
            temperature: options.temperature ?? profile.temperature,
            max_tokens: options.maxOutputTokens ?? profile.maxOutputTokens,
            messages: providerMessages(messages),
        };
        if (this.usesOpenRouter(profile)) {
            request.provider = { sort: "throughput" };
        }

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

        let modelUsed = profile.chatModel;
        const completion = await this.createChatCompletionWithValidation(
            async () => {
                const outcome = await this.createWithModelFallback(client, request, profile, options.traceContext?.traceLabel ?? "text");
                modelUsed = outcome.modelUsed;
                return outcome.completion;
            },
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
            model: modelUsed,
            rawOutput: completion.choices[0]?.message?.content || "",
            durationMs: Math.max(0, Date.now() - startedAt),
            webStatus,
            webSearchRequests,
        };
    }

    private static isRetryableProviderError(error: unknown): boolean {
        const status = (error as { status?: unknown } | null)?.status;
        if (status === 429) return true;
        if (typeof status === "number" && status >= 500) return true;
        // Credit exhaustion (402 / CreditsError) is never retryable: retrying
        // burns wall-clock and model-budgeted backoff without any chance of
        // success until the user tops up.
        return false;
    }

    private static sleep(ms: number): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    /**
     * One model request with bounded retries on 429/5xx. Client-side retries
     * stay disabled on the SDK itself; this is the only place backoff lives.
     */
    private static async requestWithProviderRetries(
        client: OpenAI,
        request: Record<string, unknown>,
        profile: ModelProfile,
        purpose: string,
    ) {
        let lastError: unknown = null;
        for (let attempt = 0; attempt <= this.RATE_LIMIT_RETRIES; attempt += 1) {
            try {
                return await ModelUsage.measure(profile, purpose, () => client.chat.completions.create(request as never, { signal: ModelUsage.signal() }));
            } catch (error) {
                lastError = error;
                if (!this.isRetryableProviderError(error) || attempt === this.RATE_LIMIT_RETRIES) {
                    throw error;
                }
                const delay = this.rateLimitBackoffMs[Math.min(attempt, this.rateLimitBackoffMs.length - 1)];
                await this.sleep(delay);
            }
        }
        throw lastError;
    }

    /**
     * Runs the request for the configured model; if a non-default model keeps
     * failing with provider errors, retries once on the default profile so a
     * flaky secondary model degrades instead of killing the turn. Local
     * endpoints never fall back: their errors are their own.
     */
    private static async createWithModelFallback(
        client: OpenAI,
        request: Record<string, unknown>,
        profile: ModelProfile,
        purpose: string,
    ): Promise<{ completion: unknown; modelUsed: string }> {
        try {
            const completion = await this.requestWithProviderRetries(client, request, profile, purpose);
            return { completion, modelUsed: profile.chatModel };
        } catch (error) {
            if (!this.usesOpenRouter(profile)) throw error;
            if (!this.isRetryableProviderError(error)) throw error;
            const profiles = readModelProfiles();
            const fallback = profiles.profiles[profiles.defaultProfile];
            if (!fallback || !this.usesOpenRouter(fallback) || profileUsesResponsesApi(fallback) || fallback.chatModel === profile.chatModel) throw error;
            const completion = await this.requestWithProviderRetries(client, {
                ...request,
                model: fallback.chatModel,
            }, fallback, purpose);
            return { completion, modelUsed: fallback.chatModel };
        }
    }

    private static async createChatCompletionWithValidation(
        fetchCompletion: () => Promise<unknown>,
        traceLabel: string,
        model: string
    ): Promise<any> {
        let lastMalformedPayload: unknown = null;

        for (let attempt = 0; attempt <= this.MALFORMED_COMPLETION_RETRIES; attempt += 1) {
            const completion = await fetchCompletion() as { choices?: unknown };
            if (Array.isArray(completion?.choices) && completion.choices.length > 0) {
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

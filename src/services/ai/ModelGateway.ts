import OpenAI from "openai";
import { getAppConfig } from "@/config/AppConfig";
import type { ModelProfile, RequestClassification, SearchPlan } from "@/types/app";

type ChatMessage = {
    role: "system" | "user" | "assistant";
    content: string;
};

type ChatOptions = {
    profile?: ModelProfile;
    temperature?: number;
    maxOutputTokens?: number;
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
        const config = getAppConfig();
        const profile = options.profile || config.modelProfile;
        const client = this.getClient();
        const completion = await client.chat.completions.create({
            model: profile.chatModel,
            temperature: options.temperature ?? profile.temperature,
            max_tokens: options.maxOutputTokens ?? profile.maxOutputTokens,
            messages,
        });

        return completion.choices[0]?.message?.content?.trim() || "";
    }

    public static async generateJson<T>(
        messages: ChatMessage[],
        fallback: T,
        options: ChatOptions = {}
    ): Promise<T> {
        const raw = await this.generateText(messages, options);
        const extracted = this.extractJson(raw);
        if (!extracted) {
            return fallback;
        }

        try {
            return JSON.parse(extracted) as T;
        } catch {
            return fallback;
        }
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
}

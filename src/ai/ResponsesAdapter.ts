import type OpenAI from "openai";
import type { ToolCall, ToolChatMessage } from "@/ai/ModelGateway";
import type { ModelProfile } from "@/shared/appTypes";
import { ModelUsage } from "./ModelUsage";
import { randomUUID } from "node:crypto";

/**
 * OpenAI Responses API adapter for providers that only expose that surface
 * (e.g. OpenCode Zen at opencode.ai/zen/v1, whose muse free models 500 on
 * /chat/completions but work on /responses).
 *
 * Converts the chat-completions message/tool shape into the Responses input
 * shape, calls client.responses.create, and converts the result back into
 * the chat-completion shape the gateway expects.
 */

type ResponsesInputItem =
    | { role: "system" | "developer" | "user" | "assistant"; content: string | Array<{ type: "input_text"; text: string } | { type: "input_image"; image_url: string; detail: "auto" | "low" | "high" }> }
    | { type: "function_call"; call_id: string; name: string; arguments: string }
    | { type: "function_call_output"; call_id: string; output: string };

export function profileUsesResponsesApi(profile: ModelProfile): boolean {
    return profile.api === "responses";
}

function toResponsesInput(messages: ToolChatMessage[]): ResponsesInputItem[] {
    const input: ResponsesInputItem[] = [];
    for (const msg of messages) {
        if (msg.role === "system") {
            input.push({ role: "system", content: msg.content });
        } else if (msg.role === "user") {
            if (msg.audio?.length) throw new Error("Audio input requires a chat-completions profile; this Responses adapter does not support it.");
            input.push({ role: "user", content: msg.images?.length ? [{ type: "input_text", text: msg.content },
                ...msg.images.map(image => ({ type: "input_image" as const, image_url: image.url, detail: image.detail ?? "auto" as const }))] : msg.content });
        } else if (msg.role === "assistant") {
            if (msg.content) input.push({ role: "assistant", content: msg.content });
            // Tool calls become flat function_call items (the assistant-role
            // tool_calls shape is rejected by Zen's upstream).
            const toolCalls = "tool_calls" in msg ? msg.tool_calls : undefined;
            if (toolCalls?.length) {
                for (const tc of toolCalls) {
                    input.push({
                        type: "function_call",
                        call_id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    });
                }
            }
        } else if (msg.role === "tool") {
            input.push({ type: "function_call_output", call_id: msg.tool_call_id, output: msg.content });
        }
    }
    return input;
}

function toResponsesTools(tools: unknown[]): unknown[] {
    return (tools as Array<{ function: { name: string; description: string; parameters: unknown } }>).map((tool) => ({
        type: "function" as const,
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
    }));
}

export interface ResponsesCallOutcome {
    content: string | null;
    toolCalls: ToolCall[];
    finishReason: string;
    usage: { promptTokens: number; completionTokens: number; totalTokens: number } | null;
    raw: unknown;
}

/**
 * Runs one Responses API call and returns a chat-completions-compatible
 * shape. Throws on provider errors like the normal path.
 */
export async function callResponsesApi(
    client: OpenAI,
    profile: ModelProfile,
    messages: ToolChatMessage[],
    options: { tools: unknown[]; maxOutputTokens: number; temperature?: number },
): Promise<ResponsesCallOutcome> {
    const body: Record<string, unknown> = {
        model: profile.chatModel,
        input: toResponsesInput(messages),
        max_output_tokens: options.maxOutputTokens,
    };
    if (profile.reasoningEffort) body.reasoning = { effort: profile.reasoningEffort, summary: "auto" };
    if (options.tools.length) {
        body.tools = toResponsesTools(options.tools);
        body.parallel_tool_calls = profile.parallelToolCalls ?? false;
    }

    const headers = profile.provider === "opencode" ? { "x-opencode-session": ModelUsage.sessionId() ?? `sophia-${randomUUID()}`, "x-opencode-client": "sophia", "User-Agent": "Sophia/5.0.0" } : undefined;
    const response = (await client.responses.create(body as never, { headers, signal: ModelUsage.signal() })) as unknown as {
        status: string;
        incomplete_details?: { reason?: string } | null;
        output?: Array<Record<string, unknown>>;
        usage?: { input_tokens?: number; output_tokens?: number; total_tokens?: number };
        error?: unknown;
    };

    let content: string | null = null;
    const toolCalls: ToolCall[] = [];
    for (const item of response.output ?? []) {
        if (item.type === "message") {
            const outContent = item.content as Array<{ type: string; text?: string }> | undefined;
            const text = (outContent ?? [])
                .filter((part) => part.type === "output_text")
                .map((part) => part.text ?? "")
                .join("\n")
                .trim();
            if (text) content = (content ? `${content}\n` : "") + text;
        }
        if (item.type === "function_call") {
            toolCalls.push({
                id: String(item.call_id ?? item.id ?? `call_${toolCalls.length}`),
                type: "function",
                function: {
                    name: String(item.name),
                    arguments: String(item.arguments ?? "{}"),
                },
            });
        }
    }

    const finishReason = response.status === "completed"
        ? (toolCalls.length ? "tool_calls" : "stop")
        : response.incomplete_details?.reason === "max_output_tokens"
            ? "length"
            : response.status || "stop";

    return {
        content,
        toolCalls,
        finishReason,
        usage: response.usage && typeof response.usage.input_tokens === "number" && typeof response.usage.output_tokens === "number"
            ? {
                promptTokens: response.usage.input_tokens,
                completionTokens: response.usage.output_tokens,
                totalTokens: response.usage.total_tokens ?? response.usage.input_tokens + response.usage.output_tokens,
            }
            : null,
        raw: response,
    };
}

import type OpenAI from "openai";
import type { ToolCall, ToolChatMessage } from "@/ai/ModelGateway";
import type { ModelProfile } from "@/shared/appTypes";

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
    | { role: "system" | "developer" | "user" | "assistant"; content: string }
    | { type: "function_call"; call_id: string; name: string; arguments: string }
    | { type: "function_call_output"; call_id: string; output: string };

export function profileUsesResponsesApi(profile: ModelProfile): boolean {
    return Boolean(profile.apiKeyEnv === "OPENCODE_API_KEY" || profile.provider === "opencode");
}

function toResponsesInput(messages: ToolChatMessage[]): ResponsesInputItem[] {
    const input: ResponsesInputItem[] = [];
    for (const msg of messages) {
        if (msg.role === "system") {
            input.push({ role: "system", content: msg.content });
        } else if (msg.role === "user") {
            input.push({ role: "user", content: msg.content });
        } else if (msg.role === "assistant") {
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
            } else if (msg.content) {
                input.push({ role: "assistant", content: msg.content });
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
        // Muse free models on Zen require the reasoning effort envelope;
        // without it upstream 500s. Harmless for other providers.
        reasoning: { effort: "minimal" as const, summary: "auto" as const },
        text: { verbosity: "medium" as const },
    };
    if (options.tools.length) {
        body.tools = toResponsesTools(options.tools);
        body.parallel_tool_calls = profile.parallelToolCalls ?? false;
    }

    const response = (await client.responses.create(body as never)) as unknown as {
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
        usage: response.usage
            ? {
                promptTokens: response.usage.input_tokens ?? 0,
                completionTokens: response.usage.output_tokens ?? 0,
                totalTokens: response.usage.total_tokens ?? 0,
            }
            : null,
        raw: response,
    };
}

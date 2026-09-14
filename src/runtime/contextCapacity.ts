import type { ToolChatMessage } from "@/ai/ModelGateway";
import { countTokens } from "@/shared/tokenizer";

const messageTokens = new WeakMap<object, { serialized: string; tokens: number }>();
let toolCache = { serialized: "", tokens: 0 };

/** Text and protocol overhead estimate. Media costs vary by provider and resolution. */
export function estimateRequestTokens(messages: ToolChatMessage[], tools: unknown): number {
    const serializedTools = JSON.stringify(tools);
    if (toolCache.serialized !== serializedTools) toolCache = { serialized: serializedTools, tokens: countTokens(serializedTools) };
    let total = toolCache.tokens + 32;
    for (const message of messages) {
        const { images, ...body } = message as ToolChatMessage & { images?: unknown[] };
        const serialized = JSON.stringify(body);
        let cached = messageTokens.get(message);
        if (cached?.serialized !== serialized) {
            cached = { serialized, tokens: countTokens(serialized) };
            messageTokens.set(message, cached);
        }
        total += cached.tokens + 8 + (images?.length ?? 0) * 2048;
    }
    return total;
}

export function availableInputTokens(contextWindow: number, outputTokens: number): number {
    return Math.max(0, Math.floor(contextWindow * 0.9) - outputTokens);
}

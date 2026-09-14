import { expect, it } from "vitest";
import { estimateRequestTokens, availableInputTokens } from "@/runtime/contextCapacity";
import type { ToolChatMessage } from "@/ai/ModelGateway";

it("accounts for tools, output reserve, images and changed message content", () => {
    const messages: ToolChatMessage[] = [{ role: "user", content: "Hi" }];
    const initial = estimateRequestTokens(messages, []);
    messages[0].content = "much more context ".repeat(1000);
    expect(estimateRequestTokens(messages, [])).toBeGreaterThan(initial);
    messages[0] = { role: "user", content: "Hi", images: [{ url: "data:image/jpeg;base64,abc" }] };
    expect(estimateRequestTokens(messages, [])).toBe(initial + 2048);
    expect(estimateRequestTokens(messages, [{ description: "tool schema ".repeat(100) }])).toBeGreaterThan(initial + 2048);
    expect(availableInputTokens(10000, 2000)).toBe(7000);
});

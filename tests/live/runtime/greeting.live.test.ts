import { afterEach, expect, it, vi } from "vitest";
import { Runtime } from "@/runtime/Runtime";
import { boundLiveModelCalls } from "./modelBudget";

afterEach(() => vi.restoreAllMocks());
const enabled = process.env.LIVE_MODEL_TESTS === "1" && Boolean(process.env.OPENROUTER_API_KEY) && process.env.OPENROUTER_API_KEY !== "test-key";
(enabled ? it : it.skip)("answers a greeting naturally through the current prompt and provider", async () => {
    const usage = boundLiveModelCalls(2);
    const result = await Runtime.answer({ question: "Olá, Sophia!", user: { id: "synthetic-smoke-user" } as never, requesterDisplayName: "Teste",
        guild: null, currentChannelId: "synthetic-smoke-dm", nativeThreadId: null, requestedWebMode: "off", trigger: "talk", replyContext: null, referencedMessage: null,
        authorize: async () => "allow", conversation: { key: "synthetic-smoke-dm", kind: "channel", trigger: "talk", replyAnchorMessageId: null, nativeThreadId: null } });
    expect(result.outcome).toBe("completed");
    expect(result.answer.trim().length).toBeGreaterThan(1);
    expect(result.answer.length).toBeLessThan(700);
    expect(result.toolRuns).toHaveLength(0);
    expect(result.answer).not.toMatch(/tool_call|function_call|runtime|memory_search/i);
    console.info(JSON.stringify({ ...usage(), answer: result.answer }));
});

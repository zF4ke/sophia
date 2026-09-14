import { expect, it, vi } from "vitest";
import { callResponsesApi, profileUsesResponsesApi } from "@/ai/ResponsesAdapter";
import { ModelUsage } from "@/ai/ModelUsage";

const profile = { chatModel: "model", embeddingModel: "embedding", temperature: 0.5, maxOutputTokens: 1000, contextWindow: 8000 };
it("selects protocol explicitly without guessing from a credential name", () => {
    expect(profileUsesResponsesApi({ ...profile, apiKeyEnv: "OPENCODE_API_KEY" })).toBe(false);
    expect(profileUsesResponsesApi({ ...profile, api: "responses" })).toBe(true);
});
it("preserves tools, image inputs and task session metadata while identifying Sophia honestly", async () => {
    const create = vi.fn().mockResolvedValue({ status: "completed", output: [{ type: "function_call", call_id: "next", name: "finish", arguments: "{}" }], usage: { input_tokens: 8, output_tokens: 2, total_tokens: 10 } });
    const run = () => callResponsesApi({ responses: { create } } as any, { ...profile, provider: "opencode", api: "responses", reasoningEffort: "high" }, [
        { role: "user", content: "Look", images: [{ url: "https://example.com/image.png" }] },
        { role: "assistant", content: "Checking", tool_calls: [{ id: "call", type: "function", function: { name: "read", arguments: "{}" } }] },
        { role: "tool", tool_call_id: "call", content: "Evidence" },
    ], { tools: [{ type: "function", function: { name: "finish", description: "Finish", parameters: { type: "object", properties: {}, required: [] } } }], maxOutputTokens: 1000 });
    const result = await ModelUsage.scope({ taskId: "task-one", actorId: "private-actor" }, run);
    await ModelUsage.scope({ taskId: "task-one" }, run);
    const [body, options] = create.mock.calls[0];
    expect(body.reasoning.effort).toBe("high");
    expect(body.input).toEqual(expect.arrayContaining([expect.objectContaining({ type: "function_call", call_id: "call" }), expect.objectContaining({ type: "function_call_output", call_id: "call", output: "Evidence" })]));
    expect(body.input[0].content[1].type).toBe("input_image");
    expect(body.input).toContainEqual({ role: "assistant", content: "Checking" });
    expect(options.headers).toEqual({ "x-opencode-session": "task-one", "x-opencode-client": "sophia", "User-Agent": "Sophia/5.0.0" });
    expect(create.mock.calls[1][1].headers).toEqual(options.headers);
    expect(result.toolCalls[0].function.name).toBe("finish");
    expect(result.usage?.totalTokens).toBe(10);
});
it("does not convert incomplete provider usage into a free attempt", async () => {
    const create = vi.fn().mockResolvedValue({ status: "completed", output: [], usage: { input_tokens: 8 } });
    const result = await callResponsesApi({ responses: { create } } as any, profile, [], { tools: [], maxOutputTokens: 10 });
    expect(result.usage).toBeNull();
    expect(create.mock.calls[0][0]).not.toHaveProperty("reasoning");
});

it("passes execution cancellation to the active Responses request", async () => {
    const controller = new AbortController();
    const create = vi.fn((_body, options) => new Promise((_resolve, reject) => {
        expect(options.signal).toBe(controller.signal);
        options.signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
        controller.abort();
    }));
    await expect(ModelUsage.scope({ taskId: "cancel-test" }, () => {
        ModelUsage.bindExecution(controller.signal);
        return callResponsesApi({ responses: { create } } as never, profile, [], { tools: [], maxOutputTokens: 10 });
    })).rejects.toThrow("cancelled");
});

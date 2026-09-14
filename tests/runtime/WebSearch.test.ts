import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { SafeWebClient } from "@/runtime/web/SafeWebClient";
import { webSearchTool } from "@/tools/webSearch";
import { ExecutionControl } from "@/runtime/ExecutionControl";

beforeEach(() => {
    vi.stubEnv("SERPER_API_KEY", "test-key");
    vi.stubEnv("BRAVE_SEARCH_API_KEY", "");
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllEnvs(); });
const context = { guild: null, currentChannelId: "dm", question: "Search" };
function respond(body: unknown) {
    return vi.spyOn(SafeWebClient, "read").mockResolvedValue({ url: "https://google.serper.dev/search", status: 200, headers: {}, body: JSON.stringify(body) });
}
it("uses Serper with header authentication and preserves distinct result snippets", async () => {
    vi.stubEnv("BRAVE_SEARCH_API_KEY", "unused");
    const read = respond({ organic: [
        { title: "Invalid", link: "http://127.0.0.1" },
        { title: "First", link: "https://example.com/a", snippet: "First snippet" },
        { title: "Duplicate", link: "https://example.com/a" },
        { title: "Second", link: "https://example.com/b" },
    ] });
    const result = await webSearchTool.capability.run(context, { query: "a & b", count: 2 });
    expect(result.data).toMatchObject({ provider: "serper", evidenceType: "search_snippets", results: [
        { title: "First", url: "https://example.com/a", snippet: "First snippet" },
        { title: "Second", url: "https://example.com/b", snippet: "" },
    ] });
    const [url, options] = read.mock.calls[0];
    expect(new URL(url).searchParams.get("q")).toBe("a & b");
    expect(new URL(url).searchParams.get("num")).toBe("2");
    expect(url).not.toContain("test-key");
    expect(options?.headers).toEqual({ "X-API-KEY": "test-key" });
    expect(read).toHaveBeenCalledOnce();
});
it("accepts an explicit empty search", async () => {
    respond({ organic: [] });
    expect((await webSearchTool.capability.run(context, { query: "nothing" })).data).toMatchObject({ results: [] });
});
it.each([{ message: "quota exhausted" }, {}, { organic: [{ title: "Private", link: "http://localhost" }] }])("does not turn provider failures into an empty search", async body => {
    const read = respond(body);
    await expect(webSearchTool.capability.run(context, { query: "query" })).rejects.toThrow();
    expect(read).toHaveBeenCalledOnce();
});
it("propagates HTTP failures without falling back to the scraper", async () => {
    const read = vi.spyOn(SafeWebClient, "read").mockRejectedValue(new Error("HTTP 429 from google.serper.dev."));
    await expect(webSearchTool.capability.run(context, { query: "query" })).rejects.toThrow("429");
    expect(read).toHaveBeenCalledOnce();
});
it("passes execution cancellation through to the web transport", async () => {
    const read = respond({ organic: [] });
    const execution = new ExecutionControl("owner", "dm");
    await webSearchTool.capability.run({ ...context, execution }, { query: "query" });
    expect(read.mock.calls[0][1]?.signal).toBe(execution.signal);
});
it("does not hide a configured Brave failure behind DuckDuckGo", async () => {
    vi.stubEnv("SERPER_API_KEY", "");
    vi.stubEnv("BRAVE_SEARCH_API_KEY", "test-key");
    const read = vi.spyOn(SafeWebClient, "read").mockRejectedValue(new Error("HTTP 401 from api.search.brave.com."));
    await expect(webSearchTool.capability.run(context, { query: "query" })).rejects.toThrow("401");
    expect(read).toHaveBeenCalledOnce();
});

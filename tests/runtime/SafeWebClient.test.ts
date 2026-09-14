import { afterEach, expect, it, vi } from "vitest";
import { SafeWebClient, isPublicAddress, publicWebUrl } from "@/runtime/web/SafeWebClient";
import { readablePage } from "@/runtime/web/ReadablePage";
import { parseSearchResults, fetchUrlTool, sourceReadTool } from "@/tools/webSearch";
import { taskStore } from "@/runtime/tasks/TaskStore";
afterEach(() => vi.restoreAllMocks());
it.each(["127.0.0.1", "10.2.3.4", "172.16.3.4", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "224.0.0.1", "168.63.129.16", "::1", "::ffff:127.0.0.1", "fc00::1", "fe80::1", "2001:db8::1", "2002:7f00:1::"])('blocks non-public address %s', address => expect(isPublicAddress(address)).toBe(false));
it("rejects URL credentials, non-web schemes and nonstandard ports", () => {
    for (const url of ["file:///etc/passwd", "https://user:secret@example.com", "http://example.com:8080", "http://2130706433", "http://[::ffff:127.0.0.1]"]) expect(() => publicWebUrl(url)).toThrow();
    expect(isPublicAddress("8.8.8.8")).toBe(true);
    expect(isPublicAddress("2606:4700:4700::1111")).toBe(true);
});
it("checks every redirect and pins each connection to the validated DNS answer", async () => {
    const resolve = vi.spyOn(SafeWebClient, "resolveHost").mockResolvedValue([{ address: "93.184.216.34", family: 4 }]);
    const request = vi.spyOn(SafeWebClient, "requestOnce").mockResolvedValue({ url: "https://example.com", status: 302, headers: { location: "http://169.254.169.254/latest/meta-data/" }, body: "" });
    await expect(SafeWebClient.read("https://example.com")).rejects.toThrow("reserved");
    expect(request).toHaveBeenCalledTimes(1);
    expect(request.mock.calls[0][1]).toEqual({ address: "93.184.216.34", family: 4 });
    resolve.mockResolvedValue([{ address: "10.0.0.1", family: 4 }]);
    request.mockClear();
    await expect(SafeWebClient.read("https://example.com")).rejects.toThrow("private");
    expect(request).not.toHaveBeenCalled();
});
it("does not forward provider credentials to a redirect", async () => {
    vi.spyOn(SafeWebClient, "resolveHost").mockResolvedValue([{ address: "93.184.216.34", family: 4 }]);
    const request = vi.spyOn(SafeWebClient, "requestOnce").mockResolvedValueOnce({ url: "https://example.com", status: 302, headers: { location: "https://other.example.org" }, body: "" }).mockResolvedValue({ url: "https://other.example.org", status: 200, headers: {}, body: "text" });
    await SafeWebClient.read("https://example.com", { headers: { "X-Subscription-Token": "test-secret" } });
    expect(request.mock.calls[1][3]).toEqual({});
});
it("keeps snippets associated with their result and extracts article content without scripts", () => {
    expect(parseSearchResults('<div class="result"><a class="result__a" href="https://example.com/a">First</a></div><div class="result"><a class="result__a" href="https://example.com/b">Second</a><p class="result__snippet">Only second</p></div>', 5)).toEqual([{ title: "First", url: "https://example.com/a", snippet: "" }, { title: "Second", url: "https://example.com/b", snippet: "Only second" }]);
    const page = readablePage({ url: "https://example.com/a", status: 200, headers: { "content-type": "text/html" }, body: '<html><head><title>A &amp; B</title></head><body><nav>Noise</nav><article><h1>Evidence</h1><p>First paragraph.</p><p>Second paragraph.</p><a href="/b">Source B</a><script>secret()</script></article></body></html>' });
    expect(page.title).toBe("A & B");
    expect(page.content).toContain("First paragraph.");
    expect(page.content).not.toContain("secret");
    expect(page.content).not.toContain("Noise");
    expect(page.links).toEqual([{ text: "Source B", url: "https://example.com/b" }]);
});
it("preserves complete task-owned sources and reads exact subsequent slices without refetching", async () => {
    const taskId = await taskStore.create({ actorId: "owner", guildId: null, channelId: "dm", conversationId: "dm", objective: "Read page" });
    const context = { taskId, actorId: "owner", guild: null, currentChannelId: "dm", question: "Read page" };
    const page = "a".repeat(500) + "b".repeat(500) + "final";
    const read = vi.spyOn(SafeWebClient, "read").mockResolvedValue({ url: "https://example.com/final", status: 200, headers: { "content-type": "text/plain" }, body: page });
    const first = (await fetchUrlTool.capability.run(context, { url: "https://example.com", max_chars: 500 })).data as { sourceId: string; nextOffset: number; content: string };
    expect(first.content).toBe("a".repeat(500));
    const second = (await sourceReadTool.capability.run(context, { source_id: first.sourceId, offset: first.nextOffset, max_chars: 500 })).data as { content: string; nextOffset: number };
    expect(second.content).toBe("b".repeat(500));
    expect(second.nextOffset).toBe(1000);
    expect(read).toHaveBeenCalledOnce();
    await expect(sourceReadTool.capability.run({ ...context, actorId: "other" }, { source_id: first.sourceId })).rejects.toThrow("does not belong");
});

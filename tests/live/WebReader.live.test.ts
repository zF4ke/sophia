import { expect, it } from "vitest";
import { SafeWebClient } from "@/runtime/web/SafeWebClient";
import { readablePage } from "@/runtime/web/ReadablePage";
it("reads a real public page through the pinned HTTP transport", async () => {
    const response = await SafeWebClient.read("https://example.com");
    expect(response.status).toBe(200);
    expect(readablePage(response).content).toContain("Example Domain");
});

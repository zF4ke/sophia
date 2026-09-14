import { expect, it, vi } from "vitest";
import { inspectRuntimeTool } from "@/tools/inspectRuntime";

it("reports configured capabilities without credentials or invented health claims", async () => {
    process.env.OPENROUTER_API_KEY = "environment-secret-marker";
    const authorize = vi.fn(async (effect: string) => effect === "none" ? "allow" as const : "ask" as const);
    const result = await inspectRuntimeTool.capability.run({ guild: null, actorId: "owner", currentChannelId: "dm", question: "Inspect environment", authorize, modelProfileName: "captured-profile" }, {});
    expect(result.data).toMatchObject({ actorId: "owner", selectedProfile: "captured-profile", permissions: { read: "allow", write: "ask", destructive: "ask" }, files: [] });
    expect(JSON.stringify(result)).not.toContain("environment-secret-marker");
    expect(JSON.stringify(result)).toContain("does not prove a service is healthy");
});

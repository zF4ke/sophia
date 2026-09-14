import { afterEach, expect, it, vi } from "vitest";
import { retrieveMessagesTool } from "@/tools/retrieveMessages";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

afterEach(() => vi.restoreAllMocks());
it("returns a cursor that can be passed back unchanged, including semantic position", async () => {
    process.env.DISCORD_TOKEN = "test";
    process.env.OPENROUTER_API_KEY = "test";
    const semantic = { lastScore: 0.5, lastCreatedTimestamp: 123, lastMessageId: "message" };
    const retrieve = vi.spyOn(UnifiedMessageRetrieval, "retrieve").mockResolvedValue({
        targetChannelIds: [], historyMessages: [], semanticMatches: [], combinedResults: [],
        continuation: { history: { perChannelOldestMessageId: { channel: "oldest" }, continuationAvailable: true }, semantic: { cursor: semantic, continuationAvailable: true } },
    } as never);
    vi.spyOn(DiscordMemoryService, "getKnownChannelsAsync").mockResolvedValue([]);
    const context = { guild: null, question: "Find messages" };
    const first = await retrieveMessagesTool.capability.run(context, { query: "*", mode: "mixed" });
    const cursor = (first.data as { cursor: { history: Record<string, string>; semantic: typeof semantic } }).cursor;
    expect(cursor).toEqual({ history: { channel: "oldest" }, semantic });
    await retrieveMessagesTool.capability.run(context, { query: "*", mode: "mixed", cursor });
    expect(retrieve.mock.calls[1][0].cursor).toEqual(cursor);
});

import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { refreshSources } from "@/runtime/refreshSources";
import { ExecutionControl } from "@/runtime/ExecutionControl";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { DiscordHistoryReader } from "@/discord/live/DiscordHistoryReader";

import { assertReadableChannels } from "@/security/SourceAccess";
import type { TurnInput } from "@/runtime/contracts";
vi.mock("@/security/SourceAccess", () => ({ assertReadableChannels: vi.fn() }));
const source = { messageId: "m", channelId: "c", guildId: "g", sourceUrl: "https://discord.com/channels/g/c/m" };
const input = () => ({ user: { id: "owner", client: { channels: { fetch: vi.fn().mockResolvedValue({ isTextBased: () => true, messages: {}, id: "c" }) } } }, guild: { id: "g" }, currentChannelId: "c", execution: new ExecutionControl("owner", "c", 5) }) as unknown as TurnInput;
beforeEach(() => {
 vi.mocked(assertReadableChannels).mockResolvedValue();
 vi.spyOn(taskStore, "hasDeletedSources").mockResolvedValue(false);
 vi.spyOn(taskStore, "currentMessageSource").mockResolvedValue(null);
 vi.spyOn(knowledgeStore, "isSourceInvalid").mockResolvedValue(false);
});
afterEach(() => { vi.restoreAllMocks(); vi.clearAllMocks(); });
it("refetches an edited message and keeps the execution alive with its new revision", async () => {
 const turn = input(); const release = turn.execution!.register();
 try {
  turn.execution!.watchSources(["m"]); turn.execution!.beforeTool();
  ExecutionControl.invalidateSource("m", { kind: "edited", url: source.sourceUrl });
  expect(turn.execution!.signal.aborted).toBe(false);
  const fetch = vi.spyOn(DiscordHistoryReader, "message").mockResolvedValue({} as never);
  vi.spyOn(DiscordMemoryService, "getStoredMessageAsync").mockResolvedValue({ content: "New content", jumpLink: source.sourceUrl + "#new", authorName: "Author", createdTimestamp: 1, attachmentsJson: "[]" } as never);
  const result = await refreshSources(turn, [source], turn.execution!.pendingSourceChanges);
  expect(result.updates[0].content).toBe("New content");
  expect(result.retained[0].sourceUrl).toContain("#new");
  expect(fetch).toHaveBeenCalledOnce();
  expect(turn.execution!.toolCalls).toBe(1);
  expect(turn.execution!.signal.aborted).toBe(false);
 } finally { release(); }
});
it("does not refetch an unchanged source", async () => {
 const read = vi.spyOn(DiscordHistoryReader, "message");
 expect(await refreshSources(input(), [source], [])).toEqual({ retained: [source], updates: [] });
 expect(read).not.toHaveBeenCalled();
});
it("keeps other sources when one is deleted and reports the specific gap", async () => {
 vi.mocked(taskStore.hasDeletedSources).mockImplementation(async ids => ids.includes("m"));
 const other = { ...source, messageId: "other", sourceUrl: source.sourceUrl + "other" };
 const result = await refreshSources(input(), [source, other], []);
 expect(result.retained).toEqual([other]);
 expect(result.updates[0].limitation).toBe("Message was deleted.");
 expect(assertReadableChannels).toHaveBeenCalledOnce();
});
it("does not call a permission error a deletion or fetch without access", async () => {
 vi.mocked(assertReadableChannels).mockRejectedValue(new Error("Missing access"));
 const read = vi.spyOn(DiscordHistoryReader, "message");
 const result = await refreshSources(input(), [source], []);
 expect(result.updates[0].limitation).toBe("Missing access");
 expect(read).not.toHaveBeenCalled();
});
it("user cancellation still stops the refresh", async () => {
 const turn = input(); turn.execution!.cancel();
 await expect(refreshSources(turn, [source], [])).rejects.toThrow("cancelled");
});

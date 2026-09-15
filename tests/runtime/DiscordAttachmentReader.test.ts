import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Collection } from "discord.js";
import { resolveDiscordAttachment } from "@/discord/live/DiscordAttachmentReader";
import { DiscordHistoryReader } from "@/discord/live/DiscordHistoryReader";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { assertReadableChannels } from "@/security/SourceAccess";
import { TaskSandbox } from "@/runtime/sandbox/TaskSandbox";
import { taskStore } from "@/runtime/tasks/TaskStore";
import type { CapabilityContext } from "@/tools/types";
vi.mock("@/security/SourceAccess", () => ({ assertReadableChannels: vi.fn() }));
const source = "https://discord.com/channels/111/222/333";
const fresh = "https://cdn.discordapp.com/attachments/222/444/screenshot.png?ex=ffffffff&hm=fresh";
const context = { actorId: "owner", guild: { id: "111" }, currentChannelId: "222", question: "Read screenshot", client: { channels: { fetch: vi.fn().mockResolvedValue({ id: "222", isTextBased: () => true, messages: {} }) } } } as unknown as CapabilityContext;
beforeEach(() => {
    vi.mocked(assertReadableChannels).mockResolvedValue();
    vi.spyOn(DiscordMemoryService, "findAttachmentMessageAsync").mockResolvedValue(source);
    vi.spyOn(DiscordMemoryService, "getStoredMessageAsync").mockResolvedValue({ jumpLink: source } as never);
    vi.spyOn(DiscordHistoryReader, "message").mockResolvedValue({ attachments: new Collection([["444", { id: "444", url: fresh, name: "screenshot.png", contentType: "image/png", size: 3 }]]) } as never);
});
afterEach(() => { vi.restoreAllMocks(); vi.clearAllMocks(); });
it("replaces an expired indexed URL using the original message", async () => {
    const result = await resolveDiscordAttachment({ ...context, attachments: [{ id: "444", name: "screenshot.png", url: fresh.replace("ffffffff", "1"), size: 3, contentType: "image/png" }] }, "444");
    expect(result.attachment.url).toBe(fresh);
    expect(result.sourceMessageIds).toEqual(["333"]);
    expect(DiscordHistoryReader.message).toHaveBeenCalledWith(expect.anything(), "333");
});
it("imports historical bytes with provenance, without requiring a current-turn attachment", async () => {
    const taskId = await taskStore.create({ actorId: "owner", guildId: "111", channelId: "222", conversationId: "222", objective: "Read screenshot" });
    const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("PNG"));
    const scoped = { ...context, taskId, requestId: "historical-attachment" };
    expect(await TaskSandbox.importAttachment(scoped, "444", "shot.png", source)).toEqual({ path: "shot.png", bytes: 3 });
    expect(String(fetch.mock.calls[0][0])).toBe(fresh);
    expect(await TaskSandbox.files(scoped)).toMatchObject([{ path: "shot.png", sourceMessageIds: ["333"], sourceChannelIds: ["222"] }]);
});
it("checks permissions before fetching the original message", async () => {
    vi.mocked(assertReadableChannels).mockRejectedValue(new Error("Missing access"));
    await expect(resolveDiscordAttachment(context, "444", source)).rejects.toThrow("Missing access");
    expect(DiscordHistoryReader.message).not.toHaveBeenCalled();
});
it("distinguishes a removed attachment from an expired URL", async () => {
    vi.mocked(DiscordHistoryReader.message).mockResolvedValue({ attachments: new Collection() } as never);
    await expect(resolveDiscordAttachment(context, "444", source)).rejects.toThrow("no longer on the original message");
});
it("rejects a CDN link where a message link is required", async () => {
    await expect(resolveDiscordAttachment(context, "444", fresh)).rejects.toThrow("original Discord message link");
});

import { afterEach, expect, it, vi } from "vitest";
import event from "@/discord/events/message/messageUpdate.event";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { ExecutionControl } from "@/runtime/ExecutionControl";
vi.mock("@/security/guildAllowlist", () => ({ isGuildAllowed: () => true }));
afterEach(() => vi.restoreAllMocks());
it("fetches partial updates without prematurely invalidating an unchanged message", async () => {
    const invalidate = vi.spyOn(ExecutionControl, "invalidateSource");
    const ingest = vi.spyOn(DiscordMemoryService, "ingestMessage").mockResolvedValue(undefined);
    const full = { id: "message", content: "Unchanged" };
    const partial = { id: "message", partial: true, guildId: "guild", fetch: vi.fn().mockResolvedValue(full) };
    await event.execute(partial as never, partial as never);
    expect(ingest).toHaveBeenCalledWith(full);
    expect(invalidate).not.toHaveBeenCalled();
});
it("reports a failed refresh as unavailable, without claiming a confirmed edit", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const invalidate = vi.spyOn(ExecutionControl, "invalidateSource");
    const partial = { id: "message", url: "https://discord.com/channels/g/c/m", partial: true, guildId: "guild", fetch: vi.fn().mockRejectedValue(new Error("Missing access")) };
    await event.execute(partial as never, partial as never);
    expect(invalidate).toHaveBeenCalledWith("message", { kind: "unavailable", url: partial.url });
});

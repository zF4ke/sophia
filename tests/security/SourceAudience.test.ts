import { afterEach, expect, it, vi } from "vitest";
import { assertReadableChannels } from "@/security/SourceAccess";
import { AccessPolicy } from "@/security/AccessPolicy";
import { assertPublicationAudience } from "@/security/DerivedSources";
import { taskStore } from "@/runtime/tasks/TaskStore";

afterEach(() => vi.restoreAllMocks());
it("does not reuse a private reply audience when publishing to another channel", async () => {
    vi.spyOn(taskStore, "requestSources").mockResolvedValue([{ messageId: "message", channelId: "source", guildId: "guild" }]);
    const guild = { id: "guild", members: { fetch: vi.fn().mockResolvedValue({ id: "owner" }) }, channels: { fetch: vi.fn() } } as any;
    guild.channels.fetch.mockResolvedValue({ guild, isTextBased: () => true, permissionsFor: (principal: unknown) => ({ has: () => principal !== "guild" }) });
    const context = { guild, actorId: "owner", question: "publish", requestId: "turn", currentChannelId: "other", privateResponse: true };
    await expect(assertPublicationAudience(context, "other")).rejects.toThrow("restricted audience");
    await expect(assertPublicationAudience(context, "source")).resolves.toBeUndefined();
});
it("permits a source in its own channel or a private reply without disclosing it to another channel", async () => {
    const guild = { id: "guild", members: { fetch: vi.fn().mockResolvedValue({ id: "owner" }) }, channels: { fetch: vi.fn() } } as any;
    const channel = { id: "source", guild, isTextBased: () => true, permissionsFor: (principal: unknown) => ({ has: () => principal !== "guild" }) };
    guild.channels.fetch.mockResolvedValue(channel);
    await expect(assertReadableChannels(guild, "owner", ["source"], { destinationChannelId: "other" })).rejects.toThrow("restricted audience");
    await expect(assertReadableChannels(guild, "owner", ["source"], { destinationChannelId: "source" })).resolves.toBeUndefined();
    await expect(assertReadableChannels(guild, "owner", ["source"], { destinationChannelId: "other", privateResponse: true })).resolves.toBeUndefined();
});

it("rechecks the originating guild grant for private cross-guild source access", async () => {
    const original = { id: "original", members: { fetch: vi.fn().mockResolvedValue({ id: "owner" }) } };
    const source = { guild: original, isTextBased: () => true, permissionsFor: () => ({ has: () => true }) };
    const client = { channels: { fetch: vi.fn().mockResolvedValue(source) } } as any;
    const current = { id: "current", channels: { fetch: vi.fn().mockResolvedValue(null) }, client } as any;
    const decide = vi.spyOn(AccessPolicy, "decide").mockResolvedValue("allow");
    await expect(assertReadableChannels(current, "owner", ["source"], { privateResponse: true, client })).resolves.toBeUndefined();
    expect(decide).toHaveBeenCalledWith("owner", original, "none");
    decide.mockResolvedValue("deny");
    await expect(assertReadableChannels(current, "owner", ["source"], { privateResponse: true, client })).rejects.toThrow("no longer granted");
});

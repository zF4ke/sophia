import { afterEach, expect, it, vi } from "vitest";
import { ArtifactStore } from "@/discord/artifacts/ArtifactStore";
import { SecurityService } from "@/security/SecurityService";
import { artifactEditTool } from "@/tools/artifactEdit";

afterEach(() => vi.restoreAllMocks());
it("persists card ownership and denies structural edits by another granted user", async () => {
    vi.spyOn(SecurityService, "initialize").mockResolvedValue();
    const admin = vi.spyOn(SecurityService, "isAdmin").mockReturnValue(false);
    const entry = { messageId: "owned-card", channelId: "channel", guildId: "guild", ownerId: "owner", expiresAt: null, specJson: JSON.stringify({ title: "Card", sections: [{ body: "Text" }] }) };
    await ArtifactStore.record(entry);
    expect(await ArtifactStore.owner(entry.messageId)).toBe("owner");
    await ArtifactStore.record({ ...entry, ownerId: "other" });
    expect(await ArtifactStore.owner(entry.messageId)).toBe("owner");
    const args = { message_id: entry.messageId, title: "Changed" };
    const other = await artifactEditTool.capability.run({ actorId: "other", guild: null, question: "edit" }, args);
    expect(other.errorMessage).toContain("owner");
    const owner = await artifactEditTool.capability.run({ actorId: "owner", guild: null, question: "edit" }, args);
    expect(owner.errorMessage).toBe("This card belongs to another server.");
    const legacy = { ...entry, messageId: "legacy-card", ownerId: undefined };
    await ArtifactStore.record(legacy);
    expect(await ArtifactStore.owner(legacy.messageId)).toBeNull();
    const denied = await artifactEditTool.capability.run({ actorId: "owner", guild: null, question: "edit" }, { ...args, message_id: legacy.messageId });
    expect(denied.errorMessage).toContain("Legacy");
    admin.mockReturnValue(true);
    const operator = await artifactEditTool.capability.run({ actorId: "operator", guild: null, question: "edit" }, { ...args, message_id: legacy.messageId });
    expect(operator.errorMessage).toBe("This card belongs to another server.");
});

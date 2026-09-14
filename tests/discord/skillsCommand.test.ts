import { afterEach, expect, it, vi } from "vitest";
import command from "@/discord/commands/tools/skills.command";
import { SkillStore } from "@/memory/SkillStore";
import { ProductStore } from "@/runtime/storage/ProductStore";
import { SecurityService } from "@/security/SecurityService";

afterEach(() => vi.restoreAllMocks());
it("restricts quarantine adoption to operators and keeps adopted content a private draft", async () => {
    await SkillStore.initialize();
    const body = { name: "Legacy", description: "Unknown ownership", instructions: "Review before using", capabilities: [], examples: [], status: "ready" };
    await ProductStore.getClient().execute({ sql: "INSERT INTO skills(id,owner_id,guild_id,channel_id,scope,status,name,revision,body_json,created_at,updated_at) VALUES('legacy-adopt',NULL,NULL,NULL,'quarantine','ready','Legacy',1,?,1,1)", args: [JSON.stringify(body)] });
    vi.spyOn(SecurityService, "initialize").mockResolvedValue();
    const admin = vi.spyOn(SecurityService, "isAdmin").mockReturnValue(false);
    const input = { user: { id: "operator" }, guildId: "g", channelId: "c", options: { getString: (name: string) => name === "id" ? "legacy-adopt" : null, getBoolean: (name: string) => name === "adopt", getInteger: () => 1 }, deferReply: vi.fn().mockResolvedValue(undefined), editReply: vi.fn().mockResolvedValue(undefined) };
    await command.execute(input as never);
    expect((await SkillStore.quarantine("g"))[0].ownerId).toBeNull();
    admin.mockReturnValue(true);
    await command.execute(input as never);
    const audience = { actorId: "operator", guildId: "g", channelId: "c" };
    expect(await SkillStore.load(audience, "legacy-adopt")).toMatchObject({ revision: 2, ownerId: "operator", status: "draft", scope: "channel" });
    expect(await SkillStore.load({ ...audience, actorId: "other" }, "legacy-adopt")).toBeNull();
    await expect(SkillStore.adoptQuarantined(audience, "legacy-adopt", 1)).rejects.toThrow("unavailable");
});

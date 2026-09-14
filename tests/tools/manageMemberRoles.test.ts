import { expect, it, vi } from "vitest";
import { manageMemberRolesTool } from "@/tools/manageMemberRoles";

it("validates the whole role selection before changing membership and returns actual IDs", async () => {
    const add = vi.fn().mockResolvedValue(undefined);
    const member = { id: "member", displayName: "Renamed member", roles: { cache: new Map(), add, remove: vi.fn() } };
    const role = { id: "role", name: "Readable label", managed: false, editable: true };
    const guild = { id: "guild", members: { fetch: vi.fn().mockResolvedValue(member) }, roles: { fetch: vi.fn(async (id: string) => id === "role" ? role : null) } } as any;
    const context = { guild, question: "change roles", actorId: "owner" };
    await expect(manageMemberRolesTool.capability.run(context, { member_id: "member", add_roles: ["role", "missing"] })).rejects.toThrow("missing");
    expect(add).not.toHaveBeenCalled();
    const result = await manageMemberRolesTool.capability.run(context, { member_id: "member", add_roles: ["role", "role"] });
    expect(add).toHaveBeenCalledOnce();
    expect(guild.members.fetch).toHaveBeenCalledWith({ user: "member", force: true });
    expect(result.data).toMatchObject({ memberId: "member", addedRoleIds: ["role"], addedRoleMentions: ["<@&role>"] });
    expect(result.summary).toContain("<@member>");
    expect(result.summary).toContain("role");
    expect(manageMemberRolesTool.capability.sideEffectLevel).toBe("destructive");
});

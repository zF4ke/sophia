import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Guild } from "discord.js";
import { SettingsService } from "@/app/SettingsService";
import { SecurityService } from "@/security/SecurityService";
import { AccessPolicy } from "@/security/AccessPolicy";
import { isGuildAllowed } from "@/security/guildAllowlist";

const guild = { id: "g1" } as Guild;
describe("access policy", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        SettingsService.update(SettingsService.getDefaults());
        vi.spyOn(SecurityService, "initialize").mockResolvedValue();
        vi.spyOn(SecurityService, "isAdmin").mockImplementation(id => id === "operator");
        vi.spyOn(SecurityService, "isModerator").mockReturnValue(false);
    });
    function grant(level: "read" | "write" | "destructive" = "write", mode: "ask" | "auto" = "ask") {
        SettingsService.update({ guildAllowlist: ["g1"], access: {
            directMessages: false, roles: [], users: [{ userId: "u1", guildId: "g1", level, mode }],
        } });
    }
    it("disables empty locations, DMs and even operator work until enabled", async () => {
        expect(isGuildAllowed("g1")).toBe(false);
        expect(isGuildAllowed(null)).toBe(false);
        expect(await AccessPolicy.decide("operator", guild, "none")).toBe("deny");
    });
    it("reads freely, asks for permitted writes and denies higher tiers or other users", async () => {
        grant();
        expect(await AccessPolicy.decide("u1", guild, "none")).toBe("allow");
        expect(await AccessPolicy.decide("u1", guild, "write")).toBe("ask");
        expect(await AccessPolicy.decide("u1", guild, "destructive")).toBe("deny");
        expect(await AccessPolicy.decide("same-display-name", guild, "none")).toBe("deny");
        grant("read", "auto");
        expect(await AccessPolicy.decide("u1", guild, "write")).toBe("deny");
        grant("destructive", "auto");
        expect(await AccessPolicy.decide("u1", guild, "destructive")).toBe("allow");
    });
    it("keeps identity grants separate from location and DM availability", async () => {
        grant();
        SettingsService.update({ guildAllowlist: ["g1", "g2"] });
        expect(await AccessPolicy.decide("u1", { id: "g2" } as Guild, "none")).toBe("deny");
        const access = SettingsService.load().access;
        access.users[0].guildId = undefined;
        SettingsService.update({ access });
        expect(await AccessPolicy.decide("u1", { id: "g2" } as Guild, "none")).toBe("allow");
        expect(await AccessPolicy.decide("u1", null, "none")).toBe("deny");
        SettingsService.update({ access: { ...access, directMessages: true } });
        expect(await AccessPolicy.decide("u1", null, "none")).toBe("allow");
    });
    it("rechecks grant revocation and disabled locations on a captured actor", async () => {
        grant();
        const authorize = AccessPolicy.forActor("u1", guild);
        expect(await authorize("write")).toBe("ask");
        SettingsService.update({ access: { ...SettingsService.load().access, users: [] } });
        expect(await authorize("write")).toBe("deny");
        grant();
        SettingsService.update({ guildAllowlist: [] });
        expect(await authorize("none")).toBe("deny");
    });
    it("rechecks expiry on an existing execution and does not turn an expired Auto grant into Ask", async () => {
        grant("write", "auto");
        const access = SettingsService.load().access;
        access.users[0].expiresAt = new Date(Date.now() + 60_000).toISOString();
        SettingsService.update({ access });
        const authorize = AccessPolicy.forActor("u1", guild);
        expect(await authorize("write")).toBe("allow");
        access.users[0].expiresAt = new Date(Date.now() - 1).toISOString();
        SettingsService.update({ access });
        expect(await authorize("write")).toBe("deny");
        expect(await authorize("none")).toBe("deny");
    });
    it("refreshes role membership and fails closed when it cannot be fetched", async () => {
        const fetch = vi.fn().mockResolvedValue({ roles: { cache: new Map([["r1", {}]]) } });
        const roleGuild = { id: "g1", members: { fetch } } as unknown as Guild;
        SettingsService.update({ guildAllowlist: ["g1"], access: { directMessages: false, users: [], roles: [
            { guildId: "g1", roleId: "r1", level: "write", mode: "auto" },
        ] } });
        const authorize = AccessPolicy.forActor("u1", roleGuild);
        expect(await authorize("write")).toBe("allow");
        expect(fetch).toHaveBeenCalledWith({ user: "u1", force: true });
        fetch.mockResolvedValue({ roles: { cache: new Map() } });
        expect(await authorize("write")).toBe("deny");
        fetch.mockRejectedValue(new Error("unavailable"));
        expect(await authorize("none")).toBe("deny");
    });
    it("retires legacy auto-write and requires an explicit Auto grant even for operators", async () => {
        grant();
        SettingsService.update({ runtime: { ...SettingsService.load().runtime, autoApproveWrites: true } } as never);
        expect(await AccessPolicy.decide("u1", guild, "write")).toBe("ask");
        expect(await AccessPolicy.decide("operator", guild, "write")).toBe("ask");
        expect(await AccessPolicy.decide("operator", guild, "destructive")).toBe("ask");
    });
    it("fails closed for unreadable settings and rejects malformed grants", () => {
        expect(() => SettingsService.update({ access: { users: [{ userId: "u1", level: "administrator" }] } } as never)).toThrow();
        vi.spyOn(SettingsService, "load").mockImplementation(() => { throw new Error("broken"); });
        expect(isGuildAllowed("g1")).toBe(false);
    });
    it("applies capability exceptions within the grant ceiling with deny then ask precedence", async () => {
        grant("write", "ask");
        const access = SettingsService.load().access;
        access.rules = [{ subject: "user", subjectId: "u1", guildId: "g1", tool: "send_message", decision: "allow" },
            { subject: "user", subjectId: "u1", tool: "delete_channel", decision: "allow" }];
        SettingsService.update({ access });
        expect(await AccessPolicy.decide("u1", guild, "write", "send_message")).toBe("allow");
        expect(await AccessPolicy.decide("u1", guild, "write", "create_poll")).toBe("ask");
        expect(await AccessPolicy.decide("u1", guild, "destructive", "delete_channel")).toBe("deny");
        access.rules.push({ subject: "user", subjectId: "u1", tier: "write", decision: "ask" });
        SettingsService.update({ access });
        expect(await AccessPolicy.decide("u1", guild, "write", "send_message")).toBe("ask");
        access.rules.push({ subject: "user", subjectId: "u1", tool: "send_message", decision: "deny" });
        SettingsService.update({ access });
        expect(await AccessPolicy.decide("u1", guild, "write", "send_message")).toBe("deny");
        access.users = [];
        SettingsService.update({ access });
        expect(await AccessPolicy.decide("u1", guild, "write", "send_message")).toBe("deny");
    });
    it("keeps permitted reads prompt-free and applies denies to operators and fresh role membership", async () => {
        grant();
        const access = SettingsService.load().access;
        access.rules = [{ subject: "user", subjectId: "u1", tier: "read", decision: "ask" },
            { subject: "role", subjectId: "restricted", guildId: "g1", tool: "send_message", decision: "deny" }];
        SettingsService.update({ access });
        expect(await AccessPolicy.decide("u1", guild, "none", "retrieve_messages")).toBe("allow");
        const fetch = vi.fn().mockResolvedValue({ roles: { cache: new Map([["restricted", {}]]) } });
        const roleGuild = { id: "g1", members: { fetch } } as unknown as Guild;
        expect(await AccessPolicy.decide("operator", roleGuild, "write", "send_message")).toBe("deny");
        fetch.mockResolvedValue({ roles: { cache: new Map() } });
        expect(await AccessPolicy.decide("operator", roleGuild, "write", "send_message")).toBe("ask");
        fetch.mockRejectedValue(new Error("offline"));
        expect(await AccessPolicy.decide("operator", roleGuild, "write", "send_message")).toBe("deny");
    });
});

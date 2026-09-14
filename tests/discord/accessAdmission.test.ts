import { beforeEach, describe, expect, it, vi } from "vitest";
import event from "@/discord/events/interaction/interactionCreate.event";
import accessCommand from "@/discord/commands/system/access/access.command";
import { SettingsService } from "@/app/SettingsService";
import { SecurityService } from "@/security/SecurityService";
import { AccessPolicy } from "@/security/AccessPolicy";

describe("Discord access admission", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        SettingsService.update(SettingsService.getDefaults());
        vi.spyOn(SecurityService, "initialize").mockResolvedValue();
        vi.spyOn(SecurityService, "isAdmin").mockImplementation(id => id === "operator");
        vi.spyOn(SecurityService, "isModerator").mockReturnValue(false);
    });
    function interaction(userId: string, commandName = "talk") {
        return { user: { id: userId }, guildId: "g1", guild: { id: "g1" }, commandName,
            isChatInputCommand: () => true, isAutocomplete: () => false, isRepliable: () => true,
            isMessageComponent: () => false, isButton: () => false, isStringSelectMenu: () => false,
            isAnySelectMenu: () => false, isModalSubmit: () => false,
            reply: vi.fn().mockResolvedValue({}) };
    }
    it("denies an ungranted user even when command visibility is public", async () => {
        SettingsService.update({ guildAllowlist: ["g1"] });
        vi.spyOn(SecurityService, "isCommandPublic").mockResolvedValue(true);
        const execute = vi.fn();
        const input = interaction("stranger");
        await event.execute(input as never, { commands: new Map([["talk", { execute }]]) } as never);
        expect(execute).not.toHaveBeenCalled();
        expect(input.reply).toHaveBeenCalledWith(expect.objectContaining({ content: "Não tens acesso à Sophia neste local." }));
    });
    it.each(["costs", "skills", "memories"])("lets an admitted non-operator open %s despite legacy visibility", async commandName => {
        SettingsService.update({ guildAllowlist: ["g1"], access: { ...SettingsService.getDefaults().access, users: [{ userId: "member", guildId: "g1", level: "read", mode: "ask" }] } });
        vi.spyOn(SecurityService, "isCommandPublic").mockResolvedValue(false);
        const execute = vi.fn();
        await event.execute(interaction("member", commandName) as never, { commands: new Map([[commandName, { execute }]]) } as never);
        expect(execute).toHaveBeenCalledOnce();
        execute.mockClear();
        await event.execute(interaction("stranger", commandName) as never, { commands: new Map([[commandName, { execute }]]) } as never);
        expect(execute).not.toHaveBeenCalled();
    });
    it("rejects foreign component clicks and autocomplete before handlers", async () => {
        const execute = vi.fn();
        for (const autocomplete of [true, false]) {
            const input = { ...interaction("stranger"), isChatInputCommand: () => false,
                isAutocomplete: () => autocomplete, isMessageComponent: () => !autocomplete,
                customId: "game:score", respond: vi.fn().mockResolvedValue({}) };
            await event.execute(input as never, { commands: new Map([["talk", { execute }]]) } as never);
            expect(autocomplete ? input.respond : input.reply).toHaveBeenCalledOnce();
        }
        expect(execute).not.toHaveBeenCalled();
    });
    it("lets an authenticated operator enable a disabled server and grant a user", async () => {
        const input = { ...interaction("operator", "access"), options: {
            getBoolean: (name: string) => name === "enable_here" ? true : null,
            getUser: () => ({ id: "u1" }), getRole: () => null,
            getString: (name: string) => name === "level" ? "write" : name === "mode" ? "ask" : null,
        } };
        const execute = vi.fn(accessCommand.execute);
        await event.execute(input as never, { commands: new Map([["access", { execute }]]) } as never);
        expect(execute).toHaveBeenCalledOnce();
        expect(SettingsService.load().guildAllowlist).toEqual(["g1"]);
        expect(await AccessPolicy.decide("u1", input.guild as never, "write")).toBe("ask");
    });
    it("does not permit a non-operator to bootstrap access", async () => {
        const execute = vi.fn();
        await event.execute(interaction("stranger", "access") as never, { commands: new Map([["access", { execute }]]) } as never);
        expect(execute).not.toHaveBeenCalled();
        expect(SettingsService.load().guildAllowlist).toEqual([]);
    });
    it("adds and removes a capability rule without silently creating or changing grants", async () => {
        const options: Record<string, string> = { capability: "send_message", rule: "deny" };
        const input = { ...interaction("operator", "access"), options: {
            getBoolean: () => null, getUser: () => ({ id: "u1" }), getRole: () => null,
            getString: (name: string) => options[name] ?? null,
        } };
        await accessCommand.execute(input as never, {} as never);
        expect(SettingsService.load().access.users).toEqual([]);
        expect(SettingsService.load().access.rules).toEqual([{ subject: "user", subjectId: "u1", guildId: "g1", tool: "send_message", decision: "deny" }]);
        options.rule = "remove";
        await accessCommand.execute(input as never, {} as never);
        expect(SettingsService.load().access.rules).toEqual([]);
    });

    it("lets an operator enable DMs and grant a global user without enabling a server", async () => {
        const input = { ...interaction("operator", "access"), guild: null, guildId: null, options: {
            getBoolean: (name: string) => ["enable_dms", "global"].includes(name) ? true : null,
            getUser: () => ({ id: "u1" }), getRole: () => null,
            getString: (name: string) => name === "level" ? "write" : name === "mode" ? "ask" : null,
        } };
        await accessCommand.execute(input as never, {} as never);
        expect(SettingsService.load().guildAllowlist).toEqual([]);
        expect(SettingsService.load().access.directMessages).toBe(true);
        expect(await AccessPolicy.decide("u1", null, "write")).toBe("ask");
        expect(await AccessPolicy.decide("stranger", null, "none")).toBe("deny");
    });
});

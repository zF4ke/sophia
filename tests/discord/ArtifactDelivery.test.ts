import { beforeEach, describe, expect, it, vi } from "vitest";
import { deliverArtifactMessage } from "@/discord/artifacts/ArtifactDelivery";
import { AccessPolicy } from "@/security/AccessPolicy";
import { ToolExecutor } from "@/runtime/ToolExecutor";
import * as gates from "@/discord/approval/ApprovalGate";
import type { AccessDecision } from "@/security/accessConfig";

describe("artifact delivery authority", () => {
    beforeEach(() => vi.restoreAllMocks());
    function setup(decision: AccessDecision) {
        const authorize = vi.fn().mockResolvedValue(decision);
        const principal = vi.spyOn(AccessPolicy, "forActor").mockReturnValue(authorize);
        const approved = vi.fn().mockResolvedValue({ approved: true });
        vi.spyOn(gates, "createApprovalGate").mockReturnValue(approved);
        const execute = vi.spyOn(ToolExecutor, "execute").mockResolvedValue({ record: { blocked: false } } as never);
        const guild = { id: "g1", channels: { cache: new Map([["c1", { isTextBased: () => true }]]) } };
        const interaction = { user: { id: "actor" }, guild, guildId: "g1", channelId: "c1", channel: { isSendable: () => true } };
        return { authorize, principal, approved, execute, interaction };
    }
    it("denies read-only users before requesting approval or executing", async () => {
        const f = setup("deny");
        await expect(deliverArtifactMessage(f.interaction as never, { channelId: "c1", content: "hello" })).rejects.toThrow("permissão");
        expect(f.approved).not.toHaveBeenCalled();
        expect(f.execute).not.toHaveBeenCalled();
        expect(f.principal).toHaveBeenCalledWith("actor", f.interaction.guild);
    });
    it("previews exact content and passes current authority to execution after approval", async () => {
        const f = setup("ask");
        await deliverArtifactMessage(f.interaction as never, { channelId: "c1", content: "hello" });
        expect(f.approved).toHaveBeenCalledWith(expect.objectContaining({ requesterId: "actor", toolArgs: { channel_id: "c1", content: "hello" } }), undefined);
        expect(f.execute).toHaveBeenCalledWith("send_message", { channel_id: "c1", content: "hello" }, expect.objectContaining({ authorize: f.authorize, actorId: "actor" }), { approved: true });
    });
    it("does not execute declined sends or resolve targets from another guild", async () => {
        const f = setup("ask");
        f.approved.mockResolvedValue({ approved: false });
        await expect(deliverArtifactMessage(f.interaction as never, { channelId: "c1", content: "hello" })).rejects.toThrow("não aprovado");
        await expect(deliverArtifactMessage(f.interaction as never, { channelId: "other-guild-channel", content: "hello" })).rejects.toThrow("neste servidor");
        expect(f.execute).not.toHaveBeenCalled();
    });
    it("auto mode executes without a prompt within the granted guild", async () => {
        const f = setup("allow");
        await deliverArtifactMessage(f.interaction as never, { channelId: "c1", content: "hello" });
        expect(f.approved).not.toHaveBeenCalled();
        expect(f.execute).toHaveBeenCalledOnce();
    });
});

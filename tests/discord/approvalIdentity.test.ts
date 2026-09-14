import { afterEach, describe, expect, it, vi } from "vitest";
import { handleApprovalInteraction, APPROVAL_MODAL_PREFIX } from "@/discord/approval/approvalInteractions";
import { BATCH_MODAL_PREFIX } from "@/discord/approval/ApprovalGate";
import { SecurityService } from "@/security/SecurityService";
import { AccessPolicy } from "@/security/AccessPolicy";
import * as gates from "@/discord/approval/ApprovalGate";

afterEach(() => vi.restoreAllMocks());

describe("approval modal identity", () => {
    it.each(["owner", "other", "revoked", "operator"])("checks requester ownership and current grants for %s", async actor => {
        vi.spyOn(SecurityService, "initialize").mockResolvedValue();
        vi.spyOn(SecurityService, "isAdmin").mockImplementation(id => id === "operator");
        const permission = vi.spyOn(AccessPolicy, "decide").mockResolvedValue(actor === "revoked" ? "deny" : "ask");
        vi.spyOn(gates, "getPendingApproval").mockReturnValue({ request: {
            requestId: "request", requesterId: actor === "revoked" ? "revoked" : "owner",
            toolName: "send_message", toolArgs: { channel_id: "c1", content: "hi" },
            description: "Send hi", sideEffectLevel: "write",
        } } as never);
        const resolve = vi.spyOn(gates, "resolvePendingApproval").mockImplementation(() => undefined);
        const read = vi.fn().mockReturnValue("Change the destination");
        const interaction = { isModalSubmit: () => true, customId: `${APPROVAL_MODAL_PREFIX}request`,
            user: { id: actor }, guild: { id: "g1" }, isFromMessage: () => false,
            fields: { getTextInputValue: read }, reply: vi.fn().mockResolvedValue({}) };
        await handleApprovalInteraction(interaction as never);
        if (actor === "owner") {
            expect(permission).toHaveBeenCalledWith("owner", interaction.guild, "write", "send_message");
            expect(resolve).toHaveBeenCalledWith("request", expect.objectContaining({ decidedBy: "owner", approved: false }));
        } else {
            expect(read).not.toHaveBeenCalled();
            expect(resolve).not.toHaveBeenCalled();
        }
    });
    it.each([APPROVAL_MODAL_PREFIX, BATCH_MODAL_PREFIX])("checks the actual submitter for %s before reading a correction", async prefix => {
        vi.spyOn(SecurityService, "initialize").mockResolvedValue();
        const policy = vi.spyOn(SecurityService, "isAdmin").mockReturnValue(false);
        const read = vi.fn();
        const reply = vi.fn().mockResolvedValue({});
        const interaction = { isModalSubmit: () => true, customId: `${prefix}request`,
            user: { id: "unauthorized" }, requesterId: "administrator",
            fields: { getTextInputValue: read }, reply, deferred: false, replied: false };
        expect(await handleApprovalInteraction(interaction as never)).toBe(true);
        expect(read).not.toHaveBeenCalled();
        expect(reply).toHaveBeenCalledWith(expect.objectContaining({ content: "Não tens permissão para alterar esta aprovação." }));
    });
});

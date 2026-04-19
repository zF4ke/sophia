import { describe, expect, it, vi } from "vitest";
import { CommandPolicyRegistry } from "@/security/CommandPolicyRegistry";
import { ACCESS_POLICY_TARGETS } from "@/security/policyTargets";
import { SecurityConfigStore } from "@/security/storage/SecurityConfigStore";

describe("CommandPolicyRegistry", () => {
    it("creates a default private command policy with role-based limits", () => {
        const registry = new CommandPolicyRegistry();

        expect(registry.getCommandConfig("ask")).toEqual({
            isPublic: false,
            rateLimits: {
                default: 5,
                admin: 10,
                moderator: 7,
            },
        });
    });

    it("persists visibility and rate-limit changes", () => {
        const registry = new CommandPolicyRegistry();
        const saveSpy = vi
            .spyOn(SecurityConfigStore, "saveCommandConfigs")
            .mockImplementation(() => undefined);

        registry.setCommandVisibility("find", true);
        registry.setCommandRateLimit("find", 3, 6, 4);

        expect(registry.getCommandConfig("find")).toEqual({
            isPublic: true,
            rateLimits: {
                default: 3,
                admin: 6,
                moderator: 4,
            },
        });
        expect(saveSpy).toHaveBeenCalledTimes(2);
    });

    it("treats message triggers like other policy targets with private-by-default limits", () => {
        const registry = new CommandPolicyRegistry();

        expect(registry.getCommandConfig(ACCESS_POLICY_TARGETS.reply)).toEqual({
            isPublic: false,
            rateLimits: {
                default: 5,
                admin: 10,
                moderator: 7,
            },
        });
    });
});

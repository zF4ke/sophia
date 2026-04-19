import type { Client } from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import { ACCESS_POLICY_TARGETS, type AccessPolicyTarget } from "@/security/policyTargets";
import type { BotClient } from "@/shared/appTypes";
import type { AdminUser, CommandConfig, ModeratorUser } from "@/security/types";

export interface AccessPanelData {
    admins: AdminUser[];
    moderators: ModeratorUser[];
    commandConfigs: Map<string, CommandConfig>;
    commandNames: string[];
    triggerNames: AccessPolicyTarget[];
}

export async function loadAccessPanelData(client: BotClient): Promise<AccessPanelData> {
    const [admins, moderators, commandConfigs] = await Promise.all([
        SecurityService.getAllAdmins(),
        SecurityService.getAllModerators(),
        SecurityService.getCommandConfigs(),
    ]);

    return {
        admins,
        moderators,
        commandConfigs,
        commandNames: [...client.commands.keys()].sort((left, right) =>
            left.localeCompare(right)
        ),
        triggerNames: Object.values(ACCESS_POLICY_TARGETS),
    };
}

export async function fetchUserLabel(client: Client, userId: string): Promise<string> {
    const cached = client.users.cache.get(userId);
    if (cached) {
        return cached.username;
    }

    const fetched = await client.users.fetch(userId).catch(() => null);
    return fetched?.username ?? userId;
}

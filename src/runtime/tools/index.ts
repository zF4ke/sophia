import type { DiscordToolName } from "@/shared/discordTools";
import { getGuildContextStrategy } from "./getGuildContext";
import { getMemberProfileStrategy } from "./getMemberProfile";
import { listGuildStructureStrategy } from "./listGuildStructure";
import { listMembersStrategy } from "./listMembers";
import { resolveChannelTargetsStrategy } from "./resolveChannelTargets";
import { resolveMemberIdentityStrategy } from "./resolveMemberIdentity";
import { retrieveMessagesStrategy } from "./retrieveMessages";
import type { ToolStrategy } from "./types";

const STRATEGY_MAP: ReadonlyMap<DiscordToolName, ToolStrategy> = new Map([
    ["retrieve_messages", retrieveMessagesStrategy],
    ["resolve_member_identity", resolveMemberIdentityStrategy],
    ["resolve_channel_targets", resolveChannelTargetsStrategy],
    ["get_member_profile", getMemberProfileStrategy],
    ["list_guild_structure", listGuildStructureStrategy],
    ["list_members", listMembersStrategy],
    ["get_guild_context", getGuildContextStrategy],
]);

export function getToolStrategy(id: DiscordToolName): ToolStrategy {
    const strategy = STRATEGY_MAP.get(id);
    if (!strategy) {
        throw new Error(`No tool strategy registered for "${id}"`);
    }
    return strategy;
}

export { type ToolStrategy } from "./types";

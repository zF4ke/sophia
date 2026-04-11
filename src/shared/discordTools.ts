export const DISCORD_TOOL_NAMES = [
    "retrieve_messages",
    "resolve_member_identity",
    "list_guild_structure",
    "resolve_channel_targets",
    "get_member_profile",
    "list_members",
    "get_guild_context",
] as const;

export type DiscordToolName = (typeof DISCORD_TOOL_NAMES)[number];
export type DiscordToolEvidenceRole =
    | "message_evidence"
    | "live_evidence"
    | "discovery_only";

export const DISCORD_TOOL_DESCRIPTIONS: Record<DiscordToolName, string> = {
    retrieve_messages:
        "Search cached Discord messages first, then automatically refresh from live Discord history when needed.",
    resolve_member_identity:
        "Resolve a member or bot in the current guild using exact ids, live guild fetches, and same-guild historical author fallback.",
    list_guild_structure:
        "List the current guild's readable channels and categories plus cached-only remembered entries.",
    resolve_channel_targets:
        "Resolve channel or category references in the current guild, including exact ids and category expansion.",
    get_member_profile: "Fetch a member profile live from Discord.",
    list_members: "List guild members live from Discord, optionally filtered.",
    get_guild_context: "Fetch live guild metadata such as name and channel counts.",
};

export const DISCORD_TOOL_EVIDENCE_ROLES: Record<
    DiscordToolName,
    DiscordToolEvidenceRole
> = {
    retrieve_messages: "message_evidence",
    resolve_member_identity: "live_evidence",
    list_guild_structure: "discovery_only",
    resolve_channel_targets: "discovery_only",
    get_member_profile: "live_evidence",
    list_members: "live_evidence",
    get_guild_context: "live_evidence",
};

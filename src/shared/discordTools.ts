export const DISCORD_TOOL_NAMES = [
    "search_messages",
    "read_message_thread",
    "read_channel_summary",
    "list_relevant_channels",
    "crawl_channel_messages",
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
    search_messages: "Search locally indexed Discord messages across channels.",
    read_message_thread: "Read stored messages around an anchor message.",
    read_channel_summary: "Summarize the stored contents of one channel.",
    list_relevant_channels: "List channels with the strongest lexical matches for a query.",
    crawl_channel_messages:
        "Fetch live messages from one readable channel, ingest them into memory, then use them in later retrieval.",
    get_member_profile: "Fetch a member profile live from Discord.",
    list_members: "List guild members live from Discord, optionally filtered.",
    get_guild_context: "Fetch live guild metadata such as name and channel counts.",
};

export const DISCORD_TOOL_EVIDENCE_ROLES: Record<
    DiscordToolName,
    DiscordToolEvidenceRole
> = {
    search_messages: "message_evidence",
    read_message_thread: "message_evidence",
    read_channel_summary: "message_evidence",
    list_relevant_channels: "discovery_only",
    crawl_channel_messages: "discovery_only",
    get_member_profile: "live_evidence",
    list_members: "live_evidence",
    get_guild_context: "live_evidence",
};

export const DISCORD_TOOL_NAMES = [
    "search_messages",
    "read_message_thread",
    "read_channel_summary",
    "list_relevant_channels",
    "get_member_profile",
    "list_members",
    "get_guild_context",
] as const;

export type DiscordToolName = (typeof DISCORD_TOOL_NAMES)[number];

export const DISCORD_TOOL_DESCRIPTIONS: Record<DiscordToolName, string> = {
    search_messages: "Search locally indexed Discord messages across channels.",
    read_message_thread: "Read stored messages around an anchor message.",
    read_channel_summary: "Summarize the stored contents of one channel.",
    list_relevant_channels: "List channels with the strongest lexical matches for a query.",
    get_member_profile: "Fetch a member profile live from Discord.",
    list_members: "List guild members live from Discord, optionally filtered.",
    get_guild_context: "Fetch live guild metadata such as name and channel counts.",
};

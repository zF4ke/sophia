export type DiscordToolEvidenceRole =
    | "message_evidence"
    | "history_evidence"
    | "semantic_evidence"
    | "live_evidence"
    | "discovery_only";

/**
 * Canonical tool name list. Per-tool definitions live in `src/tools/`.
 * Derived data (isWriteTool, TOOL_DESCRIPTIONS, etc.) lives in `src/tools/registry.ts`.
 */
const TOOL_NAMES = [
    // ── Retrieval ──
    "retrieve_messages",
    "search_messages",
    // ── Server & channel discovery ──
    "list_guild_structure",
    "get_guild_context",
    "resolve_channel_targets",
    // ── Member discovery ──
    "resolve_member_identity",
    "get_member_profile",
    "list_members",
    "get_role_info",
    // ── Thread reading ──
    "list_threads",
    "read_thread_messages",
    // ── Utilities ──
    "measure_text_length",
    "evaluate_math",
    // ── Write ──
    "create_channel",
    "create_category",
    "create_thread",
    "move_channel",
    "manage_member_roles",
    "send_message",
    // ── Destructive ──
    "clear_messages",
    "delete_channel",
] as const;

// ── Derived types and arrays ──

export type DiscordToolName = (typeof TOOL_NAMES)[number];
export const DISCORD_TOOL_NAMES = [...TOOL_NAMES] as DiscordToolName[];

/** Type-safe tool name constants. Use `T.clear_messages` instead of `"clear_messages"`. */
export const T = Object.fromEntries(
    DISCORD_TOOL_NAMES.map((n) => [n, n]),
) as { readonly [K in DiscordToolName]: K };



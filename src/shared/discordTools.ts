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
    "random_channel_message",
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
    // ── Role discovery ──
    "list_roles",
    // ── Write ──
    "create_channel",
    "create_category",
    "create_thread",
    "move_channel",
    "move_category",
    "manage_member_roles",
    "send_message",
    "edit_message",
    "create_role",
    // ── Destructive ──
    "clear_messages",
    "delete_messages",
    "delete_channel",
    "delete_role",
    "edit_channel",
    "edit_role",
    // ── Control ──
    "start_long_task",
    // ── Scratchpad ──
    "note_add",
    "note_list",
    "note_clear",
    "plan_update",
    // ── Web ──
    "web_search",
    "fetch_url",
    // ── Long-term memory ──
    "memory_search",
    "memory_remember",
    // ── Workflows ──
    "workflow_create",
    "workflow_list",
    "workflow_run",
] as const;

// ── Derived types and arrays ──

export type DiscordToolName = (typeof TOOL_NAMES)[number];
export const DISCORD_TOOL_NAMES = [...TOOL_NAMES] as DiscordToolName[];

/** Type-safe tool name constants. Use `T.clear_messages` instead of `"clear_messages"`. */
export const T = Object.fromEntries(
    DISCORD_TOOL_NAMES.map((n) => [n, n]),
) as { readonly [K in DiscordToolName]: K };


import type { DiscordToolName } from "@/shared/discordTools";

/**
 * Tool exposure — inspired by codex `ToolExposure` (Direct vs Deferred).
 * Direct tools are always sent to the model.
 * Deferred tools are hidden until discovered via `tool_search`.
 */
export type ToolExposure = "direct" | "deferred";

/** Direct tools: high-frequency, read/discovery, cheap, always visible (~16). */
export const DIRECT_TOOLS = new Set<DiscordToolName>([
    // retrieval + discovery (core loop — never deferred)
    "retrieve_messages",
    "search_messages",
    "random_channel_message",
    "resolve_member_identity",
    "resolve_channel_targets",
    "list_guild_structure",
    "get_guild_context",
    "get_member_profile",
    "list_members",
    "get_role_info",
    "list_roles",
    "list_threads",
    "read_thread_messages",
    // utilities
    "measure_text_length",
    "evaluate_math",
    // control + scratchpad (needed for long tasks)
    "start_long_task",
    "note_add",
    "note_list",
    "note_clear",
    "plan_update",
    // web + memory search (read, common)
    "web_search",
    "memory_search",
    // workflow discovery (read)
    "workflow_list",
    // discovery meta-tool itself
    "tool_search",
]);

/** Everything else is deferred until `tool_search` loads it. */
export function getToolExposure(name: DiscordToolName): ToolExposure {
    return DIRECT_TOOLS.has(name) ? "direct" : "deferred";
}

export function isDirectTool(name: DiscordToolName): boolean {
    return getToolExposure(name) === "direct";
}

export function isDeferredTool(name: DiscordToolName): boolean {
    return getToolExposure(name) === "deferred";
}

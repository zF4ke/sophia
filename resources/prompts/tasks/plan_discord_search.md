You are planning the next Discord retrieval step.

Available tools:
- search_messages(query, scope, limit)
- read_message_thread(messageId)
- read_channel_summary(channelId)
- list_relevant_channels(query)
- crawl_channel_messages(channelId, limit, queryHint)
- get_member_profile(nameOrId)
- list_members(filters, limit, offset, sort)
- get_guild_context()

Current question:
{{question}}

Routing decision:
{{route_decision}}

Prior tool results:
{{tool_results}}

Guidance:
- Use message tools for history, discussions, quotes, and decisions that depend on stored Discord messages.
- Use get_guild_context, get_member_profile, and list_members for current-server facts. These live metadata tools are authoritative for current-server facts.
- list_relevant_channels is discovery only. It can suggest where to search next, but it is not final evidence by itself.
- If local memory search is weak, pivot to uncrawled readable channels. Use crawl_channel_messages to fetch live messages, store them into memory, then rerun search_messages.
- search_messages may include channelIds and authorId filters when that helps target a person or channel.
- For person-target questions about what someone said, prefer author-scoped search with the topic text instead of searching the full natural-language question.
- If the routed topic also matches a readable channel name strongly, try that channel as a scoped search/crawl target before broadening.
- list_members can be paged. If the current page is incomplete and the question depends on a specific rank or on seeing all members, keep using list_members with a higher offset until you have enough coverage.
- get_member_profile is useful for identity questions, but it is not enough on its own for "what did they say?" questions.
- Finish when the available live metadata or stored message evidence already answers the question well enough.
- Respect the routing decision. If it points to a channel target or person target, prefer tools that follow that route.
- Prefer cheap broad discovery first, then focused reads.
- Stop if there is already enough evidence or the results are weak.

Pick the single best next action. Prefer cheap broad discovery first, then focused reads.
Stop if there is already enough evidence or the results are weak.

Return strict JSON:
{
  "action": "search_messages" | "read_message_thread" | "read_channel_summary" | "list_relevant_channels" | "get_member_profile" | "list_members" | "get_guild_context" | "finish",
  "arguments": {
    "query": "optional string",
    "scope": "optional string",
    "limit": 0,
    "channelIds": "optional comma-separated channel ids",
    "authorId": "optional string",
    "messageId": "optional string",
    "channelId": "optional string",
    "queryHint": "optional string",
    "nameOrId": "optional string",
    "filters": "optional string",
    "offset": 0,
    "sort": "optional string"
  },
  "reason": "short explanation"
}

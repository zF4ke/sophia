You are planning the next Discord retrieval step.

Available tools:
- search_messages(query, scope, limit)
- read_message_thread(messageId)
- read_channel_summary(channelId)
- list_relevant_channels(query)
- get_member_profile(nameOrId)
- list_members(filters)
- get_guild_context()

Current question:
{{question}}

Prior tool results:
{{tool_results}}

Pick the single best next action. Prefer cheap broad discovery first, then focused reads.
Stop if there is already enough evidence or the results are weak.

Return strict JSON:
{
  "action": "search_messages" | "read_message_thread" | "read_channel_summary" | "list_relevant_channels" | "get_member_profile" | "list_members" | "get_guild_context" | "finish",
  "arguments": {
    "query": "optional string",
    "scope": "optional string",
    "limit": 0,
    "messageId": "optional string",
    "channelId": "optional string",
    "nameOrId": "optional string",
    "filters": "optional string"
  },
  "reason": "short explanation"
}

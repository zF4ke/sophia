You are deciding the next step for a Discord-grounded retrieval loop.

Question:
{{question}}

Current channel id:
{{current_channel_id}}

Forced final pass:
{{forced_final}}

Explicit user mention ids:
{{explicit_user_ids}}

Explicit channel mention ids:
{{explicit_channel_ids}}

Readable channels in this server:
{{readable_channels}}

Extracted topic hint:
{{extracted_topic_hint}}

Extracted person hint:
{{extracted_person_hint}}

Referential follow-up:
{{referential_follow_up}}

Prior short-lived conversation context:
{{prior_context}}

Current grounding summary:
{{grounding_summary}}

Seeded from cached context:
{{seeded_from_context}}

Initial seeded tool run count:
{{initial_tool_runs_count}}

Already crawled channel ids:
{{crawled_channel_ids}}

Tool history:
{{tool_history}}

Choose:
- the real question intent
- the next best action
- whether message evidence is still required
- whether the final answer should be confident, best-effort, or insufficient

Question intents:
- person_identity
- person_messages
- member_list_or_ordinal
- server_context
- channel_or_topic_search
- broad_search

Allowed next actions:
- answer
- best_effort_answer
- search_messages
- get_member_profile
- list_members
- get_guild_context
- list_relevant_channels
- crawl_channel_messages

Rules:
- Explicit user mentions and explicit channel mentions are hard constraints. Respect them.
- Prior context, caches, and channel names are hints, not guarantees. Use them when they help.
- Identity questions about a person may stop on a resolved member profile.
- Questions about what someone said should usually require message evidence.
- If a topic like "silksong" is also an exact readable channel name, prefer that channel as a scoped search or crawl target.
- If local search is weak and a likely channel exists, pivot to channel discovery or crawl before giving up.
- On a forced final pass, do not choose another search tool unless there is a very strong reason. Prefer answer, best_effort_answer, or insufficient via answerConfidence.
- Use best_effort_answer when there is some real evidence but not enough for a fully confident answer.
- Use answer with answerConfidence=insufficient only when there is effectively no useful evidence.
- Keep targetText, topicText, channelHintText, and searchQuery short and literal.

Return strict JSON:
{
  "questionIntent": "person_identity" | "person_messages" | "member_list_or_ordinal" | "server_context" | "channel_or_topic_search" | "broad_search",
  "nextAction": "answer" | "best_effort_answer" | "search_messages" | "get_member_profile" | "list_members" | "get_guild_context" | "list_relevant_channels" | "crawl_channel_messages",
  "targetText": "optional short target",
  "authorId": "optional user id",
  "authorQuery": "optional author lookup string",
  "topicText": "optional topic string",
  "channelHintText": "optional channel hint",
  "channelIds": ["optional", "channel", "ids"],
  "searchQuery": "optional literal search query",
  "needsMessageEvidence": true,
  "answerConfidence": "confident" | "best_effort" | "insufficient",
  "confidence": 0.0,
  "reason": "short explanation"
}

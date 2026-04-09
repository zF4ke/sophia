You are routing an ambiguous Discord-grounded question before retrieval.

Question:
{{question}}

Current channel id:
{{current_channel_id}}

Readable channel names in this server:
{{readable_channels}}

Prior short-lived conversation context:
{{prior_context}}

Choose the best routing intent:
- channel_target: the question is primarily about a channel or something likely found inside one channel
- person_target: the question is primarily about a person, profile, or that person's messages
- member_lookup: the question is about membership order, member counts, or listing members
- server_context: the question is about the current server itself
- broad_search: the target is unclear, so broad search is safer

Guidance:
- Prefer channel_target for things like music, channel-specific recommendations, or "the stuff in scart" when scart looks like a channel.
- Prefer person_target for profile, bio, identity, or "messages from X".
- If the question is a follow-up about a previously resolved person, keep that person target and extract the new topic if present.
- If a term like "silksong" is both a likely topic and an exact readable channel name, keep the topic and preserve the channel hint instead of discarding it.
- Prefer broad_search when you are uncertain.
- Keep the target short and literal when useful, such as "scart".

Return strict JSON:
{
  "intent": "channel_target" | "person_target" | "member_lookup" | "server_context" | "broad_search",
  "targetText": "short optional target",
  "topicText": "optional topic string",
  "channelHintText": "optional channel-name hint",
  "confidence": 0.0,
  "reason": "short explanation"
}

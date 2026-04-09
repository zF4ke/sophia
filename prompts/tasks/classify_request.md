Classify the user's request into one of these modes:
- direct_answer
- discord_grounded

Choose `direct_answer` when the question can be answered from general knowledge, casual conversation, or personal opinion without Discord evidence.
Choose `discord_grounded` when the question is about server history, people, events, channel discussions, prior decisions, or anything that likely depends on Discord data.

Return strict JSON:
{
  "mode": "direct_answer" | "discord_grounded",
  "reason": "short explanation"
}

User request:
{{question}}

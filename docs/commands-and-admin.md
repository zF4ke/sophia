# Commands And Admin

## Main Conversation Surface

Use these to talk to Sophia:
- `/talk`
- mention Sophia in a message
- reply to a Sophia message

These all route to the same conversation runtime.

## Specialized Workflow

- `/find`
  Runs the specialized Discord retrieval workflow directly. It is separate from the conversation loop, but it uses the same retrieval, member-resolution, and guild-discovery primitives underneath.
  Supports:
  - `topic`
  - optional explicit `channel`
  - optional `target` for channel/category id or name
  - optional `author` for member/bot id, mention, or name

## Operator Commands

- `/nth` — read indexed historical messages
- `/index` — manage backfill and repair; `/index status` shows local retrieval state
- `/debug toggle` — enable or disable the debug panel for the current guild
- `/debug logs` — view recent model-output log entries with optional `count` and `date` filters
- `/access` — manage admin and moderator access
- `/settings` — configure runtime parameters through an interactive panel with explanations for context retention, paging, live crawl limits, and loop guardrails

`/index status` is the single status surface for local retrieval state. It exposes:
- local message/index status
- current-guild completeness data such as readable live channels/categories and cached-only remembered channels
- runtime storage status

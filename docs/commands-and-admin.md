# Commands and administration

All conversation entrypoints use the same runtime and authenticated admission checks. Command visibility never substitutes for an access grant.

## Conversation and work

| Entry | Use |
| --- | --- |
| `/talk` | Converse, attach a file without text, or explicitly resume with `task_id`. `handoff:true ephemeral:true` moves eligible inactive owned work privately. |
| Mention, reply, enabled DM | Start the same conversation runtime. Location availability and account grants still apply. |
| `/steer` | Correct active work. Supply `task_id` when selection would be ambiguous. |
| `/stop` | Cooperatively stop your execution. It does not undo an action already dispatched. |
| `/tasks` | Inspect your recent outcomes. `task_id` selects one, `file_path` downloads an owned file, and `forget:true` removes eligible retained work. |
| `/schedules` | Inspect and manage your follow-ups. |
| `/costs` | Show your own usage report across locations; use `ephemeral:true` to keep it private. |
| `/nth` | Read a position in eligible channel history. |
| `/ping` | Inspect bot latency and connection information. |

Public-task owners can add a named collaborator with `/tasks task_id:... collaborator:@person`, or remove one with `remove_collaborator:true`. The collaborator needs their own grant, can steer only the named public task, and gains no ownership of approvals or private files. Later changes require owner approval.

## Operator controls

| Command | Responsibility |
| --- | --- |
| `/access` | Enable locations, manage authenticated user/role grants, approval modes, expiry, capability/tier rules and operator access. Works for operators in disabled locations. |
| `/settings` | Select a model and shared voice; configure context/retrieval; independently switch dreaming, workspace and scheduling; inspect installation costs. |
| `/index` | Inspect, refresh, deep-backfill or reset retrieval state. Index resets preserve durable tasks, memories and settings. |
| `/superchannels` | Manage protected channels for destructive-action blocking. |
| `/memories` | Review quarantined legacy memory before private adoption. |
| `/skills` | Inspect versioned skills, assessments and quarantined procedures. |
| `/debug` | Toggle diagnostics or inspect logs. Logs can contain conversation content. |

Operator bootstrap IDs come from `BOOTSTRAP_ADMIN_IDS`, not source code. Availability and grants are governed by [access policy](access-policy.md). Legacy command-visibility administration cannot bypass admission. The obsolete `/uitest` mock panels were removed; tests exercise production builders and handlers.

## Settings tabs

The model tab reads configured profiles from `resources/models/model-profiles.json`. Muse Spark 1.3 Free uses Zen Responses; OpenRouter and local profiles remain available. Model selection does not change the runtime's ownership or approval rules.

The runtime tab configures context retention, retrieval page sizes, repeated-call protection, approval timeout and an optional whole-execution tool-call limit. Zero disables that limit. No setting adds a default total spending or duration budget.

Memory and tools controls dreaming, sandbox and scheduling independently. Disabling a service preserves its saved data. Voice selects balanced, casual or formal presentation for the single shared identity. Reset affects the selected category and preserves access rules, protected resources and storage paths.

The Custos tab is operator-only and aggregates retained provider attempts across the installation. `/costs` uses the authenticated actor's records only. Periods are rolling 24 hours, 7 days, 30 days or all retained history. Missing usage or prices remain explicit; retries and background calls are counted. Forgetting tasks removes associated cost records. See [cost accounting](cost-accounting.md).

## Approval interaction

Reads within a grant run freely. Ask decisions show the concrete operation to the requester. Only that requester can approve, deny, correct or stop the request through its controls. Being an operator does not let someone decide another person's approval.

The runtime rechecks access and rule decisions before dispatch. A changed operation needs its own approval; an expired prompt is not a denial. Destructive tiers and protected targets apply even in Auto mode. A model-supplied approval argument grants nothing.

## References

The handbook generates exact command option names from the command builders and capability schemas from the registry. `npm run docs:build` refreshes them. See [task lifecycle](task-lifecycle.md), [UI](ui.md), [access policy](access-policy.md) and [testing](testing.md) for implementation contracts.

Task discovery and steering are available through ordinary chat using `task_search` and `task_control`; `/steer`, `/stop`, and task IDs remain optional precise controls. See task-lifecycle.md for ownership and continuation checks.


## Response visibility

Ordinary conversation, settings, own costs, task summaries, steering and stop confirmations appear in the channel by default. The commands support ephemeral:true for an explicitly private reply. Public task summaries omit private work and check source audience. Task details and exports, memory and skill reviews, schedules and access reviews also default to the channel and support `ephemeral:true`. The installation-wide cost tab updates the settings message. Public controls do not relax operator or owner checks.

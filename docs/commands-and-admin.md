# Commands And Admin

## Main Conversation Surface

Use these to talk to Sophia:
- `/talk`
- mention Sophia in a message
- reply to a Sophia message

These all route to the same conversation runtime.

## Specialized Workflow

  Runs the specialized Discord retrieval workflow directly. It is separate from the conversation loop, but it uses the same retrieval, member-resolution, and guild-discovery primitives underneath.
  Supports:
  - `topic`
  - optional explicit `channel`
  - optional `target` for channel/category id or name
  - optional `author` for member/bot id, mention, or name

## Saved Workflows, Long Tasks, Artifacts

Three separate concepts, do not conflate them:

- **Saved workflows** (`workflow_create`, `workflow_list`, `workflow_run`, `workflow_delete`): named tool chains stored per guild and rerun on demand. Use `workflow_delete` to remove one (fixes the undeletable `demo-resumo` case).
- **Long tasks**: ordinary multi-step work that needs many tool calls. The model declares it with `start_long_task`, or the runtime auto-raises budgets on large explicit corpora. When unsure, the model asks the user in one sentence.
- **Artifacts** (`artifact_send`, `artifact_edit` tools, `src/discord/artifacts/`): rendered Components V2 cards from retrieved evidence, validated before send. Tabs via dropdown, pagination buttons, https link buttons, TTL auto-deletion (`ttl_days`, 0 keeps forever, default), and in-place edits by message id from the persisted spec.

## Operator Commands

- `/nth` — read indexed historical messages by position
- `/index` — manage backfill and repair; subcommands: `status`, `clear`, `repair`, `backfill_channel`, `backfill_category`
- `/debug toggle` — enable or disable the debug panel for the current guild
- `/debug logs` — browse recent model-output logs in an interactive panel
- `/superchannels` — manage protected channels used by destructive-action blocking
- `/access` — manage admin and moderator access
- `/settings` — configure runtime parameters through an interactive panel
- `/ping` — health check with round-trip latency, gateway ping, and shard info

`/index status` is the single status surface for local retrieval state. It exposes:
- local message/index status
- current-guild completeness data such as readable live channels/categories and cached-only remembered channels
- runtime storage status, including operational/checkpoint DB size on disk

## Settings Panel

`/settings` opens an interactive panel with:
- **Model tab** lists the supported runtime profiles from `resources/models/model-profiles.json`, currently GLM 5.3 Flash, GPT-OSS 120B, Ling 3.0 Flash, and Local LM Studio, with context window and pricing.
- **Runtime tab** — 10 tuning knobs for context retention, retrieval, loop guardrails, and approval (no wall-clock cap on turns)
- **Auto-Approve Writes toggle** — skip approval for non-destructive write actions
- **Reset to Defaults** — restore all settings to factory defaults

### Runtime Parameters

| Setting | Description | Presets |
|---------|-------------|---------|
| Recent Turns | Q/A pairs replayed into prompt | 3, 5, 8, 10 |
| Recent Channel Messages | Ambient context per turn | 10, 15, 25, 40 |
| Prior Tool Runs | Past tool runs scanned for evidence | 6, 12, 18, 24 |
| Prior Evidence Slice | Evidence items carried to next turn | 16, 32, 48, 64 |
| Default Retrieval Page | Rows per `retrieve_messages` call | 25, 50, 75, 100, 150 |
| Around-Message Window | Neighbors loaded around a hit | 8, 15, 25, 40 |
| Max Tool Calls | Hard cap per turn | 2–30 |
| Repeated Call Guard | Same args retry limit | 1, 2, 3 |
| Escalation Fetch Limit | Live refresh cap for retries | 50, 100, 150, 250, 400, 600, 800, 1000 |
| Approval Timeout | Admin approval wait time | 30s–5m |

## Approval System

When Sophia's model decides to call a write or destructive tool, the action goes through an approval gate before execution.

### Write Actions

Tools: `create_channel`, `create_category`, `create_thread`, `move_channel`, `manage_member_roles`, `send_message`

- An approval card is shown to the admin with:
  - Tool name and description
  - Yellow "write" badge
  - **Aceitar** (approve), **Recusar** (deny), **Recusar e corrigir** (deny with feedback), **Parar execução** (stop)
- If `autoApproveWrites` is enabled, write actions execute immediately without a card.

### Destructive Actions

Tools: `clear_messages`, `delete_messages`, `delete_channel`

- An approval card is shown with a red "destructive" badge
- After clicking **Aceitar**, a confirmation dialog appears: "Esta ação é destrutiva. Tens a certeza?"
- The admin must click **Confirmar** to execute
- Destructive actions can never be auto-approved

### Batch Destructive

When the model emits multiple destructive tool calls in a single response:

- All pending destructive actions are grouped into one batch approval card
- Actions are organized by their target Discord category
- The admin can:
  - **Aprovar tudo** — approve all actions
  - **Recusar tudo** — deny all actions
  - **Category select** — approve only actions in a specific Discord category (shown when 2+ categories)
  - **Recusar e corrigir** — open a modal to explain what should be done differently
  - **Parar execução** — stop the entire runtime

## Access Control

- **Admins** have full permissions (`["*"]`). Can add/remove other admins and moderators.
- **Moderators** have scoped permissions (`["moderate"]`). Cannot escalate themselves.
- System admin IDs are hardcoded and always loaded.
- Rate limiting applies per user per command/trigger for moderators and guests. Admins bypass rate limits entirely.
- `/access` also manages message triggers such as `@mentions` and replies to Sophia. These triggers are blocked by default for non-admins until an admin enables them. Admins can always use them.

### Command Permissions

Slash-command visibility is configurable in `/access`. Unless an admin has explicitly marked a command public, it remains private to non-admins by default. Admins can always use all commands.

| Command | Access |
|---------|--------|
| `/talk` | Admin |
| `/index` | Admin |
| `/debug` | Admin |
| `/access` | Admin |
| `/settings` | Admin |
| `/nth` | Public |
| `/ping` | Public |

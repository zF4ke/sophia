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

## Operator Commands

- `/nth` — read indexed historical messages by position
- `/index` — manage backfill and repair; subcommands: `status`, `clear`, `repair`, `backfill_channel`, `backfill_category`, `crawl_status`, `crawl_stop`, `crawl_pause`, `crawl_resume`
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
- **Model tab** — dedicated model selector with context window and OpenRouter pricing
- **Runtime tab** — context, retrieval, notebook, loop, and approval tuning
- **Compactação tab** — compaction model plus Tier-2/Tier-0 trigger controls
- **Long task tab** — elevated budgets and inline retrieval crawl controls used by `start_long_task`
- **Personalidade tab** — `default`, `mixed`, and `classic` tone selection
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
| Max Notebook Pages | Notebook pages allowed per request | 50, 100, 150, 200, 300, 500 |
| Notebook Expiry | Recent requests kept per thread before note pruning | 1, 3, 5, 8, 12, 20 |
| Max Tool Calls | Hard cap per turn | 2–30 |
| Repeated Call Guard | Same args retry limit | 1, 2, 3 |
| Latency Budget | Wall-clock timeout per turn | 10s–5m |
| Escalation Fetch Limit | Live refresh cap for retries | 50, 100, 150, 250, 400, 600, 800, 1000 |
| Approval Timeout | Admin approval wait time | 30s–5m |

### Compaction Parameters

| Setting | Description | Presets |
|---------|-------------|---------|
| Tier-2 Trigger | Context-window fraction that triggers middle-block summarisation | 60%–95% |
| Tier-0 Trigger | Context-window fraction that triggers input compaction before the call | 20%–60% |
| Tier-0 Absolute Ceiling | Absolute approximate prompt-token ceiling that forces input compaction even on huge-window models | 8k, 10k, 12k, 16k, 20k, 30k |

### Long Task Parameters

| Setting | Description | Presets |
|---------|-------------|---------|
| Max Tool Calls | Elevated cap after `start_long_task` | 50–1000 |
| Max Latency Budget | Elevated wall-clock budget after `start_long_task` | 120s–1800s |
| Evidence Floor | Minimum evidence slice preserved for long tasks | 64, 96, 128, 192, 256 |
| Inline Crawl Batches | Synchronous backfill batches `retrieve_messages` may ingest when history is partial | 0, 1, 3, 5, 10 |

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
| `/nth` | Admin by default; can be made public in `/access` |
| `/ping` | Admin by default; can be made public in `/access` |

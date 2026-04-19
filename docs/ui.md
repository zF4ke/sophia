# UI Guidelines

These rules keep Sophia’s Discord responses clear, human-facing, and easy to scan.

## Principles

- Prefer short containers over noisy walls of text.
- Keep conversational answers conversational.
- Put labels and values on the same line for simple metrics.
- Do not expose internal implementation jargon in user-facing copy.
- Avoid filler sentences that explain the UI instead of the result.
- Use multiple containers when that makes the result easier to follow.
- Keep status commands compact and readable on desktop and mobile.

## Good Defaults

- For small status summaries, use a primary container with:
  - a short title
  - one short explanatory sentence only when needed
  - inline metrics such as `Messages saved: 40`
  - one short section for related items such as channels
- Use accent colors sparingly to communicate rough state, not branding decoration.
- Prefer the language of the channel and the user when possible.

## Approval Cards

Approval cards use discord.js Components V2 (ContainerBuilder, ButtonBuilder, StringSelectMenuBuilder, ModalBuilder).

### Single-Item Cards

- Show tool name and short description
- Side-effect badge: yellow for write, red for destructive
- Buttons: **Aceitar** (Success green), **Recusar** (Danger red), **Recusar e corrigir** (Primary blue), **Parar execução** (Secondary gray)
- Timeout indicator: auto-deny after configurable `approvalTimeoutMs`
- Post-action status: ✅ approved, ❌ denied, ✏️ corrected, ⏳ timed out, 🛑 stopped

### Batch Destructive Cards

- Show count of pending actions
- List each item with tool icon + description
- Buttons: **Aprovar tudo**, **Recusar tudo**, **Recusar e corrigir**, **Parar execução**
- Category select menu: shown when actions span 2+ Discord categories (parent channels)
- "Recusar e corrigir" opens a modal asking "O que deve ser feito de diferente?"
- Post-action status: Approved / Denied / Partial (by category) / Timeout / Stopped / Corrected

### Confirmation Dialog

- Destructive actions show a confirmation after initial approval: "Esta ação é destrutiva. Tens a certeza?"
- Buttons: **Confirmar** (Danger red), **Cancelar** (Secondary gray)

## Activity Indicators

### Message Replies (mentions, direct replies)

- **Thinking**: Animated emoji reaction cycling every 2 seconds:
  `🤨 → 🧐 → 🤓 → 😎 → 🤔 → 🫡 → 😴 → 😬 → [repeat]`
- **Typing**: Discord typing indicator refreshed every 8 seconds
- Graceful degradation if bot lacks reaction permissions

### Interactions (/talk)

- Discord native "thinking" state after `deferReply` — no custom indicator needed

## Settings Panel

- Interactive panel with explicit tabs for:
  - model selection
  - runtime tuning
  - compaction tuning
  - long-task budgets
  - personality mode
- Model tab shows friendly labels, context window, and OpenRouter pricing
- Runtime tab keeps parameter tuning separated from model selection, including notebook retention controls
- Compaction tab exposes the summarizer model, Tier-2 trigger, Tier-0 trigger, and Tier-0 absolute ceiling
- Long-task tab exposes the caps applied by `start_long_task`
- Personality tab exposes `default`, `mixed`, and `classic`
- Toggle button for auto-approve writes
- Reset to Defaults button
- All labels and descriptions in Portuguese

## Index Status

- `/index status` should expose storage footprint in human units (`KB`, `MB`, `GB`) alongside the DB paths

## Avoid

- Raw IDs, cursors, or internal storage terminology in normal command output
- Decorative avatars in status cards
- Separate metric blocks when one line communicates the same thing better
- Debug-style copy such as "this panel shows the useful state"

## Commands That Should Follow This

- `/debug`
- `/index status`
- `/settings`
- approval cards
- future operator/status commands

## If You Change UI Patterns

Update these files together:

- `docs/ui.md`
- `AGENTS.md`
- the affected command implementation
- tests if command output contracts or doc references changed

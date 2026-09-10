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

## Artifacts (Model-Rendered Cards, Not Workflows)

Workflows save tool chains for reuse. Artifacts render one-off or TTL-bound interactive cards from evidence already gathered. The code seam is `src/discord/artifacts/`.

- The model sends them through the `artifact_send` tool (deferred, found via `tool_search`). Spec: `title` (max 120), optional `summary` (max 400), `sections` (1 to 8, each `body` max 3000 plus optional `heading` used as the tab label and `thumbnail_url` corner image), `gallery` (image grid, max 10 direct image URLs), `files` (max 10 file cards, re-uploaded with the message), `accent_color` (hex integer), `spoiler`, `ttl_days` (0 keeps forever, max 365, default 0).
- Media rules: image URLs must be direct links (.png/.jpg/.jpeg/.gif/.webp or known image CDNs like cdn.discordapp.com); pages are rejected by validation. Sources the model is taught to use: `get_member_profile` `avatarUrl`, attachment URLs in retrieval results, and direct links from web results.
- Full interactive surface: `accessory_button` per section (thumbnail or button, one per section), `action_rows` (buttons and all select types: string/user/role/mentionable/channel), and `handlers` — sandboxed JS per customId (`node:vm`, 100ms, no require/process/network) with `state` persistence, ephemeral `reply`, up to 3 `send`s, and card re-render helpers (`setTitle`/`setSummary`/`setSection`/`setAccent`/`setSpoiler`). `game:` customIds mutate persisted `gameState`; `action:` customIds ack plainly when no handler exists. Script errors surface to the clicker ephemerally.
- Interactivity: `navigation: {type: "select"}` renders a dropdown tab switcher, `{type: "pagination"}` renders prev/next buttons with a position indicator, `link_buttons` (max 5) render https link buttons. All controls use Components V2.
- `validateArtifactSpec` gates every send (shape, char budget of 3900, https-only link URLs). `buildArtifactComponents` renders per view state; `ArtifactSession` owns the in-memory interaction session (15 min, controls render disabled afterwards).
- `ttl_days` persists to the `artifacts` table and a boot-time sweep (`ArtifactStore.sweepExpired`) deletes expired cards. Interaction sessions do not survive a reboot; cards stay readable with controls disabled.
- Cards persist by default (`ttl_days` 0). `artifact_edit` updates an existing card in place from the persisted spec (title, summary, sections, navigation, buttons, TTL); only the passed fields change.
- Keep titles short, one idea per section, plain words over decoration.
- Persistent cards suit digests the server rereads; a short TTL suits throwaway confirmations.

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

- Interactive panel with explicit tabs for model selection and runtime tuning
- Model tab shows the supported profiles from the canonical model catalog, with friendly labels, context window, and OpenRouter pricing.
- Runtime tab keeps parameter tuning separated from model selection
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

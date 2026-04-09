# UI Guidelines

These rules exist to keep Sophia's Discord responses clean, readable, and human-facing instead of debug-shaped.

## Principles

- Prefer short cards over noisy walls of text.
- Put labels and values on the same line for simple metrics.
- Do not add decorative thumbnails or images unless they carry useful information.
- Do not expose internal implementation jargon in user-facing copy.
- Avoid filler sentences that explain the UI instead of the result.
- Prefer plain channel or status lines over bullet-heavy formatting.
- Keep status commands compact and easy to scan on desktop and mobile.

## Good Defaults

- For small status summaries, use a single `ContainerBuilder` with:
  - a short title
  - one short explanatory sentence only when needed
  - inline metrics such as `Mensagens salvas: 40`
  - one short section for related items such as channels
- Use accent colors sparingly to communicate rough state, not branding decoration.
- Prefer Portuguese copy for bot-facing product text unless a command already has a strong reason to stay in English.

## Avoid

- Raw IDs, cursors, or internal storage terminology in normal command output
- Decorative avatars in status cards
- Separate metric blocks when one line communicates the same thing better
- Debug-style copy such as "this panel shows the useful state"

## Commands That Should Follow This

- `/ping`
- `/cache`
- `/index status`
- `/debug`
- Other future status or diagnostics commands

## If You Change UI Patterns

Update these files together:

- `docs/ui.md`
- `AGENTS.md`
- the affected command implementation
- tests if command output contracts or doc references changed

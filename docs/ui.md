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

## Avoid

- Raw IDs, cursors, or internal storage terminology in normal command output
- Decorative avatars in status cards
- Separate metric blocks when one line communicates the same thing better
- Debug-style copy such as "this panel shows the useful state"

## Commands That Should Follow This

- `/debug`
- `/index status`
- future operator/status commands

## If You Change UI Patterns

Update these files together:

- `docs/ui.md`
- `AGENTS.md`
- the affected command implementation
- tests if command output contracts or doc references changed

# Commands And Admin Operations

## Core Commands

- `/ask` general assistant entrypoint with automatic Discord retrieval when needed
- `/find` targeted evidence search across indexed messages
- `/context` grounded answer path
- `/talk` conversational command path
- `/nth` fetch the Nth indexed historical message in a channel
- `/debug` control the global live debug mode for Sophia responses

## Admin Commands

- `/index` manage backfill, repair, clear, and status
- `/access` open the admin-only access panel for admins, moderators, and command policies
- `/cache` inspect local memory statistics

## Relevant Files

- `src/discord/commands/tools/`
- `src/discord/commands/system/`
- `src/discord/commands/system/access/access.command.ts`
- `src/discord/commands/system/access/`
- `src/security/`

The `access` command is a single command that opens a navigable component-based panel. Buttons, select menus, and the limits modal are handled from the files under `src/discord/commands/system/access/`.

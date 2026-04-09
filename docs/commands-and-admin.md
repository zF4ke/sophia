# Commands And Admin Operations

## Core Commands

- `/ask` general assistant entrypoint with automatic Discord retrieval when needed
- `/find` targeted evidence search across indexed messages
- `/context` grounded answer path
- `/talk` conversational command path
- `/getmessage` fetch the Nth indexed historical message in a channel

## Admin Commands

- `/index` manage backfill, repair, clear, and status
- `/access` manage admins, moderators, and command policies
- `/cache` inspect local memory statistics

## Relevant Files

- `src/discord/commands/tools/`
- `src/discord/commands/system/`
- `src/discord/commands/system/access/access.command.ts`
- `src/discord/commands/system/access/`
- `src/security/`

The `access` command is a nested command entrypoint loaded recursively, with helper files colocated under `src/discord/commands/system/access/`.

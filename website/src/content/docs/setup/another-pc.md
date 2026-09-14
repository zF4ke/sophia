---
title: Move to another PC
sidebar:
  order: 3
---

Install Node, Git and Docker on the destination, then clone the repository and run `npm ci`. Build the sandbox locally with `npm run sandbox:build`; containers and images are not copied by a Git checkout.

## Start fresh

Create a new `.env` from the example and supply your own credentials and operator IDs. Run `npm run doctor`. Enable the intended guilds and grants after starting the bot. Do not run both copies with the same bot token at the same time.

## Keep Sophia's durable state

Stop Sophia on the source PC before archiving.

```sh
npm run state -- backup <new-archive-directory>
```

Transfer the archive privately. On the destination, restore into an empty storage directory.

```sh
npm run state -- restore <archive-directory> <empty-storage-directory>
```

Set `SOPHIA_STORAGE_ROOT` to that destination if it is not the normal `storage` folder. Recreate `.env` separately. The archive excludes credentials and rebuildable message indexes. Read [backup and migration](../../build/cleanup-migration/) for the exact retained files and validation.

## Verify machine-specific settings

Check the local model endpoint and model ID, Docker image name, enabled guilds, grants, protected resources and any absolute paths. Keep active SQLite databases on a local disk rather than syncing them between running PCs. Interrupted tasks pause on startup; inspect before explicitly resuming.

Run `npm run doctor`, `npm run check`, and the local sandbox suite before starting normal work. Provider tests are opt-in and may spend credits. The docs website has a separate lockfile; use `npm ci` inside `website` to build it on another PC.

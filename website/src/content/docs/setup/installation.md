---
title: Install on your PC
sidebar:
  order: 1
---

Use a supported Node.js release and a Linux Docker engine. Node 22.19 or newer supports both the bot and documentation toolchain. The bot toolchain supports Node 20.19 or Node 22.12 and newer. Windows uses Docker Desktop with its WSL 2 backend; macOS uses Docker Desktop; Linux can use Docker Engine.

## Prepare the project

```sh
git clone https://github.com/zF4ke/sophia.git
cd sophia
npm ci
```

Copy `.env.example` to `.env` using your file manager or shell. Fill in `DISCORD_TOKEN`, `OPENROUTER_API_KEY`, `OPENCODE_API_KEY` for Zen, and your verified Discord user ID in `BOOTSTRAP_ADMIN_IDS`. Never commit `.env`.

Sophia still needs OpenRouter for embeddings, even when the main model is local or uses Zen. Runtime settings belong in `storage/settings.json`. Secrets belong in `.env`.

## Connect Discord

Create a bot in the Discord Developer Portal, enable the privileged Message Content and Server Members intents used by Sophia, and invite it with the bot and application commands scopes. Grant the channel access and action permissions you intend to use. Discord's role hierarchy still applies.

```sh
npm run sandbox:build
npm run doctor
npm start
```

Startup registers application commands. Start only one Sophia process per storage directory. Run `/access enable_here:true` in the target guild, then grant an account with `/access user:@person level:write mode:ask`. New installations enable no guilds and disable DMs. The doctor may report availability as needing attention until this step is complete.

Select the main model in `/settings`. Open `/settings` → Custos for installation-wide usage. Start with a mention or `/talk`.

## Check the workspace

With Docker running and the image built, run the opt-in local container suite.

PowerShell:

```powershell
$env:LIVE_SANDBOX = '1'
npm run test:sandbox
```

macOS/Linux:

```sh
LIVE_SANDBOX=1 npm run test:sandbox
```

These tests execute local containers. They do not call a model or send Discord messages. For other verification, see [testing](../../build/testing/).

## Work on the website

```sh
cd website
npm ci
npm run dev
```

The site runs on localhost. `npm run docs:build` from the repository root builds the static site. Search indexing is available in a production build and preview. Engineering pages are generated from `docs/`; edit that source rather than generated copies under `website/src/content/docs/build/`.


For separate bot and website terminals, the static preview, and deployment to zf4ke.github.io/sophia/, see [run and publish the website](../website/).

---
title: Run and publish the website
sidebar:
  order: 5
---

The website is a static Astro Starlight handbook. It runs separately from the Discord bot. Running the website does not start Sophia or require her API keys.

## Start locally

From the repository root, install the dependencies once:

```sh
npm ci
npm --prefix website ci
```

Then start the development server:

```sh
npm run docs:dev
```

Open the address Astro prints, normally `http://127.0.0.1:4321/`. Leave this terminal running. Use another terminal with `npm start` for Sophia. If a preview already occupies port 4321, stop it before launching another server or use the alternate port Astro prints.

## Preview the built site

Stop the development server, then run:

```sh
npm run docs:build
npm run docs:check
npm run docs:preview
```

This preview includes the generated search index. The check validates links, anchors, local assets, and simulated conversation structure.

## Publish to GitHub Pages

The intended public address is [zf4ke.github.io/sophia](https://zf4ke.github.io/sophia/).

The repository's Pages source should be **GitHub Actions**. The workflow in `.github/workflows/pages.yml` builds and publishes `website/dist` after relevant changes reach `master`. It also supports a manual workflow run. Its Pages configuration supplies the public origin and `/sophia` base path, including for navigation, search, images and the favicon.

Local changes do not update the public site until they are committed, pushed to the publishing branch, and the deployment succeeds. The bot continues to run on your PC; GitHub Pages hosts only the documentation.

To test that deployment path locally in PowerShell:

```powershell
$env:DOCS_SITE = 'https://zf4ke.github.io'
$env:DOCS_BASE = '/sophia'
npm run docs:build
npm run docs:check
Remove-Item Env:DOCS_SITE
Remove-Item Env:DOCS_BASE
```

Rebuild with `npm run docs:build` afterward to return to the default local preview paths.

## Edit the right source

User and setup chapters live in `website/src/content/docs/`. Engineering chapters are copied from `docs/` during each build. Tool and command references are generated from the code, so their schemas stay current.

The original icon lives in `assets/logo.svg`. The header and simulated Sophia avatar reuse it; the build also copies it to the public favicon. Purple accents live in `website/src/styles/custom.css`, with separate light and dark values for readable text. The README uses the same original icon and links to the repository's actual license.

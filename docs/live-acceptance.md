# Live acceptance record

Validation on 14 September 2026 uses synthetic model scenarios, isolated test stores and one explicitly authorized private Discord channel. Tests are evidence for the behavior exercised, not a blanket guarantee.

## Model runtime

Muse Spark 1.3 Free through OpenCode Zen passed all six live runtime scenarios: greeting, quoted creative instruction, correction, exact member identity, grounded image reference and scoped category research. The live calls use Sophia's Responses adapter with honest client identity and task session metadata. Log: `storage/v5-muse-live.log`.

The earlier GLM profile also passed all six. GPT-OSS generated unsupported service claims in a separate evaluation and remains a model-specific limitation. No model receives an automatic claim of equivalent behavior just because it is selectable.

## Discord fixtures

The user authorized creating a new private test channel. The exact guild, channel and message IDs remain in the local acceptance receipt rather than the public handbook.

Plain message send/edit, production card payloads for both pages/navigation states, and a native poll passed REST acceptance. All fixture messages were deleted; exact-ID fetches returned Unknown Message. The empty private channel remains. Local receipt: `storage/v5-discord-acceptance-recheck.json`.

The user waived further Discord visual interaction checks. Real gateway clicks and Discord desktop/mobile rendering remain unverified, and are not blockers to finishing this requested local implementation.

## Containers

Docker's Linux engine is now responding and `npm run sandbox:build` built `sophia-sandbox:5`. No rejected destructive repair was retried. The previous engine-unavailable condition is resolved.

The opt-in container suite passed execution in Python, JavaScript and shell, analysis-library imports, workspace file transfer, read-only root enforcement, no outbound connection, no host credentials, symlink exclusion and per-operation timeout. The eight real card-script tests passed state mutation, replies, contextual inputs, send bounds, rendering helpers, errors, restricted script globals and oversized-state preservation. Logs: `storage/v5-live-sandbox.log` and `storage/v5-artifact-live.log`.

Video inspection additionally exercises a generated clip through the real owned workspace and timestamped decoder. Audio provider transcription remains covered by deterministic byte/provenance tests rather than a live speech-quality evaluation.

## Documentation

The Starlight site separates user guides, installation and developer contracts. It generates references for all 78 capabilities and all registered commands. Builds check missing capability mappings; link validation checks generated HTML and anchors at both root and GitHub Pages subpaths.

Browser checks verified chapter navigation, production search for dreaming, phone-width reading/menu behavior and both color themes. Preview runs locally; this work does not deploy the site.

## Repeat the checks

Run `npm run check` for TypeScript and deterministic tests. Set `LIVE_SANDBOX=1` and run `npm run test:sandbox` for local Docker tests. Run `npm run test:live -- tests/live/ArtifactScript.live.test.ts` for card scripts. Model tests require the explicit `LIVE_MODEL_TESTS=1` switch, a selected `LIVE_MODEL_PROFILE` and locally loaded credentials; see [testing](testing.md). `npm run docs:build` and `npm run docs:check` validate the website.

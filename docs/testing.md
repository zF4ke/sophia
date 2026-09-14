# Testing

The current deterministic suite covers task handoff/deletion, source-derived publication, `/nth` audiences, grant expiry, memory quarantine adoption, ordered mutation dispatch and fair model scheduling. Live tests use synthetic Discord fixtures and explicit test-only call/output/cost bounds. `LIVE_MODEL_PROFILE` selects an existing configured profile for those tests; it does not alter deployment settings.

The current Muse Spark 1.3 Free profile and the earlier GLM profile passed greeting, creative revision, conversational correction and grounded research checks. GPT-OSS embellished a service description with unsupported features, including after a stricter prompt. Keep that failure visible when assessing model suitability. Actual container and card-script checks now pass. The user waived further Discord visual interaction checks; real gateway clicks and Discord layouts are not claimed as verified.

Live and deterministic suites share isolated storage setup. Only deterministic setup blocks `fetch`, `http.request` and `https.request`. Run the public-web smoke check with `npm run test:live -- tests/live/WebReader.live.test.ts`; it does not call a model. Container integration remains in the opt-in artifact script tests and requires the built sandbox image.

Sophia uses two test lanes on purpose.

## Fast Deterministic Tests

Run:

```bash
npm run test
```

This is the default engineering suite and the one used by `npm run check`.

It keeps the model mocked so the runtime can be validated deterministically:
- capability wiring
- capability effect and approval-policy consistency
- workflow steps cannot bypass normal runtime execution
- workflow deletion removes saved workflows and reports missing names
- larger evidence contexts do not create or raise task execution limits
- artifact spec validation (title, sections, TTL bounds) before render
- artifact guard rejects finishes while a failed artifact_send is unsent, until fixed or corrections exhausted
- interrupted and legacy crawler queue recovery
- current-guild discovery
- retrieval contracts
- conversation continuity
- debug rendering
- storage behavior
- runtime execution control, explicit optional limits, and repeated-call guard
- follow-up evidence reuse across turns
- continuation input gating (no automatic cross-turn exclusions without explicit continuation intent)
- strict scoped empty-result recovery retries in retrieval
- insufficient-confidence no-speculation guard behavior

These tests are not trying to prove that a remote model will always choose the perfect tool sequence. They are there to prove that the runtime, tool contracts, and orchestration stay correct.

## Live Model Tests

Run:

```bash
LIVE_MODEL_TESTS=1 LIVE_MODEL_PROFILE=museSpark13Zen node --env-file=.env node_modules/vitest/vitest.mjs run -c vitest.live.config.ts tests/live/runtime
```

The live suite is opt-in and expensive. It is excluded from `npm run check`.

The greeting smoke test uses at most two model calls; synthetic research scenarios use at most six per case. Each caps output at 2,048 tokens and checks a $0.10 estimated ceiling using configured prices with a retry allowance. This is an estimate, not a provider billing cap. Tests use isolated storage and synthetic Discord fixtures. The greeting smoke test passed on the configured GLM 5.3 Flash profile on 14 September 2026; that does not validate every model or actual Discord delivery.

It hits the real model API to validate:
- tool-calling behavior with real prompts
- synthesis behavior with real prompts
- runtime stories where the model must stay conversational after tool use

The live suite still mocks non-model infrastructure when needed so the test isolates model behavior instead of depending on live Discord or production storage state.

## Why Both Lanes Exist

Using only mocked tests is not enough, because the model can still behave badly even when the runtime contract is correct.

Using only live tests is also not enough, because:
- they are expensive
- they are slower
- they are less deterministic
- they are a poor fit for low-level contract validation

The intended workflow is:
1. keep `npm run check` fast and strict
2. use `npm run test:live` before merging prompt/runtime changes that depend on real model behavior
3. promote important regressions from live findings into deterministic coverage when possible

## Current Live Coverage

The live suite currently checks:
- greeting stays conversational (no unnecessary tool calls)
- current-guild research is selected for explicit member/channel references
- exact member-id answers stay natural
- grounded retrieval answers stay natural after tool execution

Add new live stories when changing:
- the agent loop prompt
- tool-composition behavior
- answer style after grounded retrieval

Recent deterministic regression additions are in:
- `tests/runtime/runtimeUserStories.test.ts` (follow-up can answer from reused prior evidence without rerunning tools)
- `tests/runtime/planning.test.ts` (fresh follow-ups do not auto-apply cursor/exclusions; model tool choice is honored without pre-flight redirects; ambiguous members profiled before generic channel discovery)
- `tests/runtime/approvalGate.test.ts` (single-item approval card creation for write/destructive tools; batch approval card creation for grouped destructive calls; destructive confirmation dialog flow; auto-approve for write tools when enabled; timeout handling; correction modal feedback; Discord category grouping in batch cards)
- `tests/discord/UnifiedMessageRetrieval.test.ts` (strict scoped retry recovers hidden rows)
- `tests/discord/DiscordLiveService.test.ts` (list_members pagination with offset, no duplicate members across pages, default page size of 20, get_member_profile returns enriched fields: joinedAt, accountCreatedAt, premiumSince, pending)

## Manual Integration Tests

These are the canonical real-server test cases used to validate end-to-end behavior. They require a live bot instance and a real guild.

### The Two Drennan Problem

Tests ambiguous identity detection, profile escalation, and decision-quality synthesis.

**Prompt:**
> Olá, @Sophia. Eu estou meio confuso tem dois seres com o vulgo Drennan e agora que lascou de vez. Como eu posso saber qual é o verdadeiro? Eu preciso mandar uma mensagem muito importante para o verdadeiro, é uma questão de vida ou morte, então, por favor, não erre! @Sophia

**Expected behavior:**
- Fetches a profile for each member sharing the display name "Drennan" (not just one)
- Compares roles, join dates, and activity
- Returns a specific recommendation: which one is the real one and why
- Does not ask the user to choose manually or return a generic answer

**What this covers:**
- Ambiguous display name detection after `list_members` or `resolve_member_identity`
- `get_member_profile` called multiple times with different identifiers in one research pass
- Profile evidence includes join date and account age for disambiguation
- Synthesis makes a concrete decision instead of deflecting

---

### The Link of February 9th

Tests temporal scoping, multi-tool composition, and follow-up continuity.

**Prompt:**
> @Sophia 9 de fevereiro o openrosen mandou um link do youtube de uma musica para o canal de comandos. ele chegou a referir de quem era?

**Expected behavior:**
- Resolves `openrosen` as a member
- Resolves `comandos` as a channel
- Retrieves messages with author + channel + time bounds (after Feb 9, before Feb 10)
- Answers from message evidence: yes, and quotes what was said about the song

**Follow-up:**
> kkkkkkkk vdd eu lembro. ele diz que foi assistir bem "ecletico" sei la como escreve.

**Expected follow-up behavior:**
- Reuses the prior evidence (no retrieval rerun needed)
- Responds naturally to the conversational aside ("cetico") without going into research mode

**What this covers:**
- Date parsing from natural Portuguese
- Member + channel + time scope composition
- Evidence-based answer without hallucination
- Follow-up continuity without unnecessary tool re-execution

---

### The 4 Services Problem

Tests category discovery, structure inspection, scoped retrieval, and response quality under time pressure framing.

**Prompt:**
> @Sophia estou com pressa e precisava de um resumo do que tem nos canais do Serviços. quero uma descrição de cada serviço

**Expected behavior:**
- Resolves "Serviços" as a category
- Inspects the guild structure to enumerate child channels
- Retrieves scoped messages from each child channel
- Returns a brief description of each service based on message content, not invented summaries

**What this covers:**
- Category → guild structure → scoped retrieval chain
- Multiple `retrieve_messages` calls with different channel scopes
- Model decides the composition without being pre-wired to run the sequence
- Budget behavior under multi-channel retrieval

## Test isolation

Recovery tests cover pending approval invalidation on restart, retained decisions and steering, failure to persist an approval, correction persistence failure, single-winner resume claims, restored objective/corrections, and refusal to resume unresolved mutations. Historical approvals are never used as execution authority.

Task evidence tests cover owner/location isolation, adjacent tasks, duplicate record rejection, context selection without deletion, full source/cursor preservation across resets and restarts, and rejection of writes to interrupted tasks. Runtime tests verify same-task evidence reuse, exclusion of channel-wide tool history, and a pause when result persistence fails.

Action receipt regressions cover persistence before dispatch, duplicate invocation rejection, unauthorized settlement, revocation during persistence, lost success receipts, malformed thrown values, and restart recovery of unsettled attempts. Artifact tests distinguish correctable argument validation from an unknown send outcome that pauses without another model call.

Workspace regressions cover same-channel task isolation, forged goal IDs, concurrent note sequence allocation, plan version increments, persistence through index resets/restarts, and owner-only state export. Runtime stories verify that a neighboring task's open goals do not trigger continuation and that unfinished goals leave the task paused.

Task tests verify owner/location visibility, atomic terminal transitions, duplicate completion rejection, restart interruption, preservation across runtime-data resets, and shared task IDs across continuation legs. No restart test executes a provider or Discord mutation.

Steering regressions cover actor/channel ownership, ambiguous executions, ordered corrections, late submissions during finalization, an in-flight model response, corrections during a tool call, valid skipped-call transcripts, and a correction arriving during the final permission check of approved work.

Each test file gets a unique storage directory. Deterministic tests block unmocked fetch calls, and four workers prevent cold-import contention. Use the separate live configuration for real-provider checks. Regression tests cover 205 tool calls, 21 continuation legs, settings migration, cancellation ownership, and approval-modal identity.

Access regressions cover disabled locations, explicit user/role grants, forced role refresh, grant revocation, action tiers, requester-owned decisions, denied autocomplete/component/message entry, operator bootstrap, and scripted card sends through approval and execution. A model-supplied `approved` argument cannot satisfy an executor approval requirement.

Source-update regressions cover newer text, stale and unseen older crawl revisions, late note/file writes, automatic memory invalidation, edited-text reversions, and deleting every revision. Access tests cover capability/tier rule precedence, grant ceilings, named collaborator revocation and private-task exclusion, and owner approval after restored collaborator steering. Approval controls reject other users even when they are operators.

On 2026-09-14, `storage/v5-final-expanded-check.log` recorded 114 files and 492 passing deterministic tests, including TypeScript validation. `storage/v5-muse-live.log` recorded all six current Muse Spark live scenarios passing. The earlier GLM pass and GPT-OSS failure remain in their respective logs. Discord REST fixtures passed and were removed. See [live acceptance](live-acceptance.md) for container and website checks.

## Local container and documentation checks

Set `LIVE_SANDBOX=1` and run `npm run test:sandbox` after `npm run sandbox:build`. This uses Docker but sends no Discord message or provider request. Card script integration runs with `npm run test:live -- tests/live/ArtifactScript.live.test.ts`.

For model tests, load `.env` locally and set `LIVE_MODEL_TESTS=1` and `LIVE_MODEL_PROFILE=museSpark13Zen`. For example, `node --env-file=.env node_modules/vitest/vitest.mjs run -c vitest.live.config.ts tests/live/runtime` runs the synthetic runtime scenarios. Test-only request/output/spending bounds do not modify deployment settings.

Install website dependencies with `npm ci` inside `website`. From the repository root run `npm run docs:build` and `npm run docs:check`. The reference generator rejects missing/duplicated capability mappings. HTML checks validate internal links and anchors; set `DOCS_BASE=/sophia` to test the Pages deployment path. Browser review verifies navigation, search, mobile layout and themes separately.

## Conversational task controls

The task-control suite covers owner/location isolation, private-task filtering, exact instruction forwarding, selected cancellation, completed-task reopening, saved workspace recovery, and rejection of unresolved action outcomes. Message adapter tests cover an explicit Sophia mention while replying to another person's message. Opt-in real-model checks in `tests/live/runtime/taskControl.live.test.ts` cover natural steering, stopping, ambiguous target clarification, and continuation with saved notes and a prior answer.

## Voice checks

`tests/live/runtime/personality.live.test.ts` exercises the selected real model with an English correction, Portuguese banter, a technical explanation, and a code sample containing an intentional em dash. The checks reject em/en dashes in original prose and habitual closing offers while preserving exact code. Passing examples are behavioral evidence, not a guarantee that every model will follow every style instruction on every turn.


## Release regressions

The client intent test verifies DM subscription. Message-event tests cover SendMessagesInThreads separately from parent-channel sending. Access admission covers ordinary skill and memory inspection. Cancellation tests use the real single/batch approval transports and verify removal of pending controls, settled records and subsequent task resumability. Provider tests verify that active HTTP calls receive the execution's AbortSignal. Settings tests serialize every tab and reject duplicate custom IDs.

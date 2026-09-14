# Sophia v5 implementation plan

The v5 implementation replaces the 4.5 prototype with one model-led runtime, one identity, explicit authority and durable work. This file tracks the delivered scope. Engineering contracts live in the linked documents; the handbook contains the user examples.

## Delivered scope

| Area | Implementation | Contract |
| --- | --- | --- |
| Conversation | Unified loop, coherent voice, replies/mentions, attachment-only input, exact authorship, no deterministic intent router | [Agent loop](agent-loop.md) |
| Identity and access | One identity; authenticated Discord IDs; explicit guild/DM availability; expiring user/role grants; Ask/Auto tiers; capability rules; protected resources | [Access](access-policy.md) |
| Sustained work | Durable tasks, plans/notes/goals, steering, cancellation, named public-task collaborators, owned resume and private handoff | [Tasks](task-lifecycle.md) |
| Recovery | Startup pause, persisted mutation receipts, uncertain-outcome verification, no blind action replay | [Tasks](task-lifecycle.md) |
| Research | Shared crawler, exact cursors, coverage, immutable collections, deduplication, index refresh/deep backfill, source versions | [Memory and indexing](memory-indexing.md) |
| Web | Search, safe URL fetch, readable sources and further source reads | [Web research](web-research.md) |
| Workspace | Isolated Python/JavaScript/shell, owned files/import/export/publication, image and timestamped video inspection, configured audio transcription | [Workspace](workspace-and-media.md) |
| Learning | Scoped memories, corrections/forgetting, idle dreaming, versioned skills, quarantine/assessment, workflow compatibility | [Skills](skills.md) |
| Follow-through | Explicit schedules, timezones, current authorization and quiet conditional monitoring | [Scheduling](scheduling.md) |
| Product UI | Task progress/details/stop with public command defaults, durable cards/revisions/click state, native polls, consistent settings and owned approvals | [UI](ui.md) |
| Models | Muse Spark 1.3 Free through Zen Responses; retained OpenRouter/local selection; explicit protocol/reasoning; shared capacity | [Architecture](architecture.md) |
| Costs | Provider-attempt records, own /costs report, operator Custos tab, date/model breakdowns, honest unknown costs | [Accounting](cost-accounting.md) |
| Portability | Validated settings, migration backups, offline state transfer, startup instance guard, Docker image build and doctor command | [Migration](cleanup-migration.md) |
| Documentation | Starlight website, separate user/setup/developer paths, simulated conversations, generated schemas/options/defaults and feature coverage | [README](../README.md) |

Tasks have no default total call, duration or spending cap. Concurrency, context capacity and per-operation bounds remain. An explicitly configured tool-call limit can pause work, with zero disabling it. Usage accounting adds no execution budget.

## Review and validation

The expanded deterministic suite passed 511 tests in 119 files on 14 September 2026. The selected Muse Spark 1.3 Free profile passed six live conversation and research scenarios. The earlier GLM default also passed those six scenarios. GPT-OSS retained a documented unsupported-claim failure in its separate evaluation.

Docker's Linux engine became available and the sandbox image built successfully. Real container execution, isolation, transfer, timeout and card-script checks passed. The latest media and documentation results are recorded in [live acceptance](live-acceptance.md). Read the [adversarial review](adversarial-review.md) for findings and their fixes.

Discord REST acceptance passed the authorized message-edit, two-page card and poll fixtures. All fixture messages were deleted. The user explicitly waived further Discord visual interaction work. Gateway clicks and Discord desktop/mobile rendering are not claimed as verified.

## Cleanup decisions

Removed legacy personas and duplicate prompts, the old landing/workflow website, obsolete mock UI command, duplicate Claude handbook, old budget/status settings and redundant runtime paths. Stable workflow tool IDs remain compatibility entrypoints into skills. Durable user data is preserved through explicit migrations and backups; index cleanup does not delete memories or task records.

Vector search, automatic multi-agent teams, a remote plugin marketplace, distributed workers, broad host access and self-editing runtime remain deliberate deferrals. They were not required for this local v5 architecture. The model can discover and compose the existing capabilities without another agent runtime.

## Handoff

Local implementation and validation do not publish the branch, start a production bot or enable guild access. The Pages workflow builds the new website when merged to master. On another PC, follow the handbook's installation/transfer guide, recreate credentials and build the sandbox image locally. Only one process may own a storage root.

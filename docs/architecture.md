# Architecture

Sophia has one conversational runtime, one Discord retrieval pipeline and one identity. Conversations select the current exchange; durable tasks own work. Availability and source audiences determine where the identity can act and what it can disclose.

## Runtime boundaries

| Owner | Responsibility |
| --- | --- |
| ConversationAdapter | Authenticated Discord event, requester, reply context, attachments, response visibility and delivery binding |
| AccessPolicy | Explicit enabled locations, user/role grants, action tiers and current Ask/Auto decisions |
| Runtime / ExecutionControl | Native tool loop, task binding, cancellation, steering, continuation and context capacity |
| ToolExecutor | Capability validation, caller-owned approvals, serialized guild mutations, publication audiences and receipts |
| TaskStore | Task ownership, handoff, outcomes, working state, approvals, evidence, files and retention |
| ModelUsage | Provider-attempt accounting across task work, compaction, transcription and dreaming |
| DiscordHistoryReader | Serialized page ingestion and checkpoint commits shared by refresh and backfill |
| UnifiedMessageRetrieval | Chronological and relevance retrieval, cursors, local index and live escalation |
| KnowledgeStore / DreamingService | One identity, audience-scoped memories, revisions, forgetting and idle consolidation |
| SkillStore | Versioned procedures, private learned drafts, retirement and provenance |
| TaskSandbox / ContainerSandbox | Owned file operations and isolated code/media processing |
| ScheduleStore / Scheduler | Explicit future work, occurrence claims, current authorization and delivery outcomes |
| ArtifactStore | Durable interactive card specification, state, serialization and retryable expiry cleanup |

One startup listener owns each canonical storage root. A second process fails before login or interrupted-task recovery. There are no distributed workers or automatic task replays.

## Persistence

| Store | Contents | Reset policy |
| --- | --- | --- |
| `storage/settings.json` | Versioned runtime preferences, availability, grants and protected channels | Atomic updates; category resets preserve unrelated configuration |
| `storage/tasks.sqlite` | Durable task ledger, evidence, files, sources and usage | Preserved by index/runtime reset |
| `storage/knowledge.sqlite` | Identity, memories, source tombstones and dream queue | Preserved by index/runtime reset |
| `storage/products.sqlite` | Cards, schedules, skills and immutable skill revisions | Preserved by index/runtime reset |
| Configured operational SQLite database | Indexed Discord messages and operational traces | Retrieval tables are reconstructible; reset migrates legacy durable records first |

Settings and provider profiles are validated before use. Secrets remain in environment-backed credentials. OpenRouter supplies remote model access; configured compatible endpoints can override a profile. No model IDs are embedded in fallback source code. Configured modalities describe which inputs a model accepts; configuration alone is not a health check.

## Instructions and evidence

External prompts define runtime policy. The system prompt receives authenticated identifiers and machine configuration. Names, quotations, remembered facts, page text, media and prior results enter separately as source data. Compaction preserves the original request and complete tool-call/result groups; a summary cannot become system authority.

Core tools are initially visible and other schemas are discovered through `tool_search`. `inspect_runtime` exposes relevant configuration, current permission decisions and owned file metadata without credentials. Every schema follows the strict provider contract in AGENTS.md.

## Actions and recovery

Reads run within the current grant. Changes require Ask approval or an applicable Auto grant within the permitted action tier. Protected targets remain protected. The executor saves mutation intent before dispatch and concrete output after success. An uncertain outcome pauses work and blocks replay. Owners can record what they verified before explicitly resuming; this attestation remains distinct from automated verification.

Cancellation and steering stop unstarted work at checkpoints. An external request already in flight can still complete. Approval and availability are rechecked before execution. Tasks have no default call, duration, continuation or note quota; an operator can set an explicit tool-call limit. Individual network/container operations remain bounded.

## Media, learning and follow-through

Code and decoders run inside the same unprivileged, network-disabled container boundary with no host mounts. Task files persist outside the disposable container. Image input, sampled video and source-labelled audio transcription remain distinct forms of evidence. Publication uses the normal action policy.

Dreaming processes delivered eligible conversations while idle. It can save supported memories and draft a procedure from demonstrated tools. Drafts remain private; publishing or promoting a procedure uses ordinary write approval. Scheduled work has an explicit owner, destination and timezone. Conditional checks may record a quiet completed result; failures remain reportable.

## Release work

Source invalidation, shared crawl ownership, supported postcondition verification, provider accounting, private handoff, retention and migration review are implemented and tested. Remaining acceptance work and model-specific failures are tracked in [the v5 plan](v5-plan.md). Docker integration is blocked by the local engine startup failure; no host execution fallback exists. Real Discord desktop/mobile behavior still needs direct acceptance testing.

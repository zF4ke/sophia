# Adversarial review

Review scope: the v5 runtime and its ownership, authorization, source provenance, recovery, model transports, container boundary and documentation contracts. This is a record of concrete checks and findings, not a claim that tests prove the absence of defects.

## Findings fixed

| Finding | Fix and evidence |
| --- | --- |
| Operator status could decide another actor's approval. | Requester-owned component checks, fresh authorization at dispatch, and approvalIdentity tests. |
| Edited sources could leave old facts in notes, memories, files or a late collection page. | Versioned source URLs, monotonic edit metadata, invalidation guards on late writes and transactional stale-page rejection. SourceInvalidation tests cover revisions, reversions and deletion. |
| A collaborator correction could cause later writes under the owner's Auto mode. | Persisted collaborator provenance forces owner approval for subsequent changes, including after resume. |
| A new external default could be sent to OpenRouter after another model failed. | Fallback accepts only a compatible OpenRouter profile. It cannot cross endpoint/protocol boundaries. |
| Cached endpoint clients ignored the selected credential. | Client cache identity includes the endpoint and credential. A missing declared key fails before dispatch. |
| Responses protocol selection depended on the name of a secret variable. | Explicit profile API and reasoning settings replace inference from the environment variable name. |
| Responses conversion dropped assistant text accompanying tool calls. | Preserve both text and function-call items, covered by adapter regression. |
| Partial Responses usage became zero tokens. | Incomplete usage remains unknown; cost-panel tests distinguish missing data from known zero prices. |
| The new personal costs command inherited legacy operator-only command visibility. | Admit `/costs` after the ordinary account/location grant check. An event-level regression verifies both an admitted non-operator and a stranger. |
| The old website and Claude handbook described a different permission model. | One Starlight website, canonical engineering docs and an AGENTS pointer replace duplicated legacy content. Old mock UI routes are removed. |

## Final release pass, 14 September 2026

The release pass found and fixed these additional defects:

- Runtime settings repeated a component custom ID. Every settings tab now has a serialized uniqueness check.
- Channel context, prior turns and fallback evidence dropped timestamps. Model context now retains UTC timestamps and source links where available, with explicit guidance for incomplete latest-message searches.
- The gateway omitted the DirectMessages intent, so enabled DMs could not reach the message handler. The client now subscribes to DM messages.
- Thread replies checked SendMessages instead of SendMessagesInThreads. The handler now uses the permission for its destination type.
- Skills and memories inherited the obsolete administrator-only command check. Authorized users can inspect their own records; legacy adoption still requires an operator.
- Cancellation reached queued model calls but not active HTTP requests or approval waits. Both provider transports now receive the execution signal. Pending single and batch approvals resolve without approval, remove their controls and release the durable pending record on cancellation.

The final validation includes the deterministic suite, ten Muse Spark personality/conversation/task-control scenarios, four real sandbox scenarios and eight real card-script scenarios. Full dependency audits, including development tools, reported zero known vulnerabilities for the bot and handbook after updating Vitest and tsx. A separate clean npm ci verified the lockfile. Linux CI exposed a Windows-only path assertion; it now uses path.join. The website is built and checked at both `/` and `/sophia/` before publication.

## Boundary checks

Access tests include disabled locations, forged principals, expired/revoked grants, explicit-deny precedence, role lookups, approval argument forgery and source-audience checks. Task tests include unknown action outcomes, duplicate dispatch, restart pause, competing resume, retention, private handoff and collaborator revocation. These remain part of `npm run check`.

Real Docker tests execute Python, JavaScript and shell, round-trip files, import the analysis libraries, attempt a root write and outbound network connection, check absent credentials, exclude symbolic links, bound a non-finishing operation and decode timestamped video through an owned task. All four scenarios passed on 14 September 2026. The initial root-write assertion expected PermissionError; the actual read-only filesystem correctly raised EROFS. The assertion now accepts only the expected denial errno values. Eight separate real card-script scenarios also passed.

The current Muse Spark 1.3 Free profile passed all six live conversation/research scenarios through Sophia's runtime. The client identifies itself as Sophia and sends a real task ID for session routing. These synthetic cases do not establish all possible model behavior. The earlier GPT-OSS unsupported-claim failure remains a model-specific limitation.

## Documentation checks

The handbook build checks coverage of all 78 capability IDs, generates current command options and configuration defaults, and validates internal links and anchors. User examples are explicitly simulated. Developer contracts have a separate navigation group. Browser checks cover reading navigation and production search; responsive/theme checks are recorded with the final validation run.

## Remaining limits

The user waived Discord visual interaction checks after successful REST fixtures. Real gateway clicks and Discord desktop/mobile layouts are not claimed as verified. The tests do not establish sandbox resistance to an unknown container-runtime vulnerability or eliminate model hallucination. Deployment, command registration and access activation remain separate from local implementation checks.

# Feature user stories

These stories describe implemented behavior and its acceptance boundaries. The tool registry owns capability names and effect tiers; this document avoids a second hand-maintained catalog.

## Conversation and identity

A greeting receives a short conversational reply without mandatory tools or a task report. Creative revision preserves the user's requested meaning and treats quoted instructions as source text. An explicit correction overrides an earlier name or presentation preference.

Sophia has one identity across enabled locations. The same owner can explicitly hand off inactive work with `/talk task_id:... handoff:true ephemeral:true`. Notes, evidence and files follow the task, subject to current source access. Other users do not inherit control from being in the same channel.

## Access and approvals

An unauthorized user cannot start model work through slash commands, mentions, replies, autocomplete or card controls. Empty availability enables no guilds; DMs require explicit availability. Identity comes from the authenticated Discord ID.

Reads run freely within a grant. Changes use Ask or Auto within the granted tier. Expired grants and revoked roles stop later calls. Role membership changes and channel moves require the access-sensitive tier. Protected channels remain blocked in the executor even after approval.

An owner can choose a stricter Ask mode for one task. Approval records include the exact request and decision. Expiry is distinct from denial. A saved decision never authorizes replay after restart. Grouped approval covers the listed operations, not arbitrary future actions.

## Sustained work and recovery

A progressing task can pass the former 25/200 tool-call and continuation limits. An optional configured call limit pauses incomplete work; the model cannot raise it. Context compaction, queueing and operation resource limits preserve the task objective.

`/steer task_id` and an owned reply to an active task redirect pending work. A correction is saved before acknowledgement. Already dispatched actions are not described as cancelled. Guild mutations are ordered, while unrelated guilds and reads proceed. Model calls share capacity fairly across actors; waiting conversation takes precedence over dreaming.

Restart pauses interrupted tasks. Unknown action outcomes block resume. Supported postcondition verification can establish a deletion or exact message result without retrying it; partial observations remain unknown. Owner attestation is labeled separately from automated verification.

## Discord and web research

Member and channel names resolve to concrete IDs. Historical membership is distinguished from current membership. Channel names and topics do not establish what people posted or which services are available; those summaries require relevant messages.

Retrieval exposes exact cursors and honest index coverage. An unindexed or partially indexed channel can be refreshed or queued for deeper history. Failed enqueueing is not reported as queued. One history reader owns page ingestion and checkpoint updates. Refreshing recent messages cannot erase deep-history progress.

A task collection retains filters, deduplicated message IDs, revisions, cursors and coverage through restart. Reading old results, exporting collections or opening derived files rechecks source access. Deleted messages invalidate dependent evidence and interrupt active readers. `/nth` applies the same source-audience restriction before revealing indexed text.

Web search discovers sources; opened pages provide evidence. Source handles let the model read more without repeatedly downloading a page. DNS, redirects and response-size limits apply to downloads. Search snippets and page instructions do not become authoritative facts or policy.

## Workspace and media

Attachment-only requests reach the runtime. Supported profiles receive images. Video inspection returns sampled frames with timestamps. Audio transcription uses a declared compatible model and saves a transcript linked to the inspected clip.

Code runs in the isolated Docker workspace with no host mounts, credentials or outbound network. Files belong to a task and persist across calls. Users can inspect and download files privately; publication checks the destination audience and approval policy. Failed container startup does not trigger host execution.

## Memory and skills

Idle dreaming can remember useful facts and presentation preferences without an explicit reminder from the user. Private conversations produce owner-private facts and skill drafts. Only whitelisted presentation preferences follow a user into public contexts.

Memories have source links, revisions and forgetting controls. Deleted sources invalidate derived content. Legacy facts with unknown provenance stay quarantined until an operator adopts an exact revision as private memory.

A saved procedure is loaded into the ordinary tool loop. Loading it does not run or approve its steps. Learned drafts require an assessment tied to their exact content and accessible completed-task evidence before promotion. The assessment is a trace review, not an independent execution test. Retirement prevents automatic recreation of the same learned procedure.

## Cards and follow-through

A card has an owner, message ID, source links, saved specification and revisions. Stale edits fail before overwriting a newer revision. Clicks and edits serialize. Inaccessible sources block later redisplay. State-save failure after sending preserves the message ID and pauses the task without sending a duplicate.

Native polls use Discord's poll API. Card scripts run in the isolated workspace and propose external actions through ordinary approval. Expiry cleanup retries failed Discord deletions and removes local revisions only after the message is gone.

Schedules retain an owner, destination, timezone and notification condition. They recheck permission after restart. An unchanged conditional check can stay quiet; failures and required action remain reportable. Uncertain delivery pauses rather than replaying blindly.

## Inspection, deletion and migration

Private task exports include plans, notes, evidence, action receipts, approval history and model usage. Inactive resolved tasks can be forgotten without deleting neighboring work. Separately remembered facts, approved skills, original Discord messages and operator backups have independent retention.

Settings reset only the selected category. Migrations back up durable data before changing it. Offline state archives preserve identity and access rules, exclude credentials and reconstructible indexes, and restore only into empty storage. Startup recovery rechecks authority instead of replaying actions.

## Validation limits

Deterministic tests exercise these boundaries with synthetic Discord objects and isolated databases. Bounded live tests cover selected conversational and research behavior. GPT-OSS has a retained failure for unsupported service details; the default GLM profile passed that stricter case. Actual Docker execution and real Discord desktop/mobile usability still require acceptance testing. Passing a mock test is not evidence that those integrations passed.

## Describe the task in chat

A user mentions Sophia with “focus on scheduling in the event comparison.” Sophia finds the owner's eligible task and forwards the actual instruction. “Stop discussing prizes” redirects the scope; “stop the event comparison” cancels the selected execution. If two tasks match, Sophia asks which one instead of asking for an ID or choosing arbitrarily.

For “revise the welcome message from earlier,” Sophia can reopen a completed task with its saved notes, files and previous answer. Task IDs and slash commands remain optional controls. Discovery is restricted to owned work in the current location; private handoff remains explicit. A restored answer is source material, never an instruction or permission, and is removed from working context if its sources are no longer eligible.

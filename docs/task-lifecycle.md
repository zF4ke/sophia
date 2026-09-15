# Durable task records

Attachment metadata persists with the owning task. Explicit resume restores those attachments. `sandbox_import` refreshes signed URLs by fetching the original Discord message, located through the local attachment index or an explicit `message_url`. If neither identifies the source, Sophia needs the original message link. Deleted attachments and inaccessible messages remain unavailable. Workspace files survive continuation and restart. `/tasks task_id:...` lists saved file paths in the task export, and `file_path:...` downloads one owned file, up to 8 MiB. Use `ephemeral:true` for private delivery.

Each top-level conversation run creates one durable task record. A task has an authenticated owner, original guild/channel, conversation key, objective, timestamps and outcome. Continuation legs share its ID. The outer run records the final outcome after continuation finishes; individual legs cannot prematurely complete the task.

`/tasks` shows your ten most recent tasks in the current channel through an ephemeral response. The lookup checks the authenticated Discord user ID and the current guild/channel. Administrative access does not make another user's task history appear in this view. Task IDs are identifiers, not permission tokens. Continuation verifies ownership, location, process session and running status before entering the model loop.

## Storage and restart behavior

`storage/tasks.sqlite` contains `tasks`, `task_events`, `task_notes`, `task_goals`, `task_actions`, and `task_tool_runs`. It is separate from disposable retrieval/runtime data, so index repair and runtime-data resets cannot erase this state. Task creation and its started event commit together. A final outcome and event also commit together, and only the owning process can finish its running task. Competing or repeated completion attempts cannot overwrite a terminal outcome.

This release assumes one bot process per deployment. On startup, records left running by another process session become paused with reason `interrupted`. Completed, cancelled, failed and already-paused records retain their outcomes. The startup pass never calls a model or tool and never retries an external action. Invalid databases raise an error rather than being replaced with an empty database.

Outcomes describe runtime execution: `completed`, `paused`, `cancelled`, or `failed`. Pausing after a restart means the outcome of any in-flight external action is unknown. The ledger does not prove that an action failed or that its effects were undone.

## Plans, notes and goals

`TaskWorkspace` binds every working-state query to a verified active task. Request and conversation IDs remain provenance; they cannot broaden a query. Notes and goals have stable sequence IDs across continuation turns. Plan updates increment a persisted version atomically. A finished model turn with open, in-progress or blocked goals leaves the task paused if continuation cannot complete them.

`note_list` returns the active task's plan, notes and goals by default. The legacy `include_thread_history` flag never includes a neighboring task. The runtime reloads the task plan each round and recent task notes each continuation leg, without guessing whether a prior answer's wording means work should resume. Only this task's goals drive continuation. New task workspaces have no arbitrary total note/goal-count cap; individual tool content limits still apply.

Saved plans and notes enter the transcript as assistant working context, before the user request, rather than as system instructions. Their persistence does not give quoted source text authority over the requester or policy.

`/tasks task_id:<id>` exports an owned task's saved plan, goals and notes through an ephemeral attachment in its original channel. Interrupted tasks can be inspected but cannot be mutated or resumed through this command. Historical request-scoped notes/goals are not automatically imported: their old thread-wide access rules do not establish task ownership. Direct legacy callers without a task binding retain their old storage adapter; all Discord runtime work supplies a verified task ID.

## Action receipts

For task-bound write and destructive capabilities, `ToolExecutor` saves the validated arguments and invocation ID before entering the handler. If this write fails, the handler never runs. Authorization, cancellation and steering are checked again after persistence. A superseded attempt is `skipped`. Schema failures before dispatch remain ordinary correctable errors.

A validated tool return becomes a `succeeded` receipt containing the actual result and resource IDs. This records what the handler returned; it does not independently verify every postcondition. An exception, error result, invalid output or lost receipt after dispatch makes the outcome `unknown` and pauses the conversation without another model call. This is conservative because existing handlers do not distinguish precondition failures from errors after partial effects. Startup also marks unsettled attempts from the previous process `unknown`.

The same task and invocation ID cannot dispatch twice. This is not an exactly-once guarantee across fresh invocations or retries at external services. Unknown actions require checking their effects before another attempt. Automatic replay remains disabled. Scripted artifact sends also create owned tasks and receipts. `/tasks task_id:<id>` includes action arguments, status and results in the private export.

## Evidence across continuation

Continuation keeps the original user request intact. Unfinished goals and the previous answer enter as source context; they cannot grant approval. Progress uses measured state through the existing notifier, with no extra model request just to phrase a status line. There is no fixed delay or default continuation count. An explicitly supplied continuation limit leaves unfinished work paused.

`task_tool_runs` stores full tool records, including arguments, output, source identifiers and retrieval cursors. The runtime writes each normal or forced retrieval result before proceeding. Unknown mutations remain in `task_actions` and pause immediately. A failed evidence write also pauses without repeating the tool; a successful mutation receipt remains valid even if the later evidence write fails.

Continuation reloads only the owning task's tool records. Context limits select recent records for the model; they do not delete earlier records. Operational channel-wide tool history remains available for diagnostics but no longer supplies a task's prior evidence. This is an execution log, not a reusable result cache, and new requests still execute their requested tools.

Evidence survives runtime/index resets and process restarts. `/tasks task_id` includes a separate `task-evidence.json` attachment when records exist. Lookup checks the authenticated owner, original location and current source-channel access. Deleted sources redact stored tool results and remove dependent collection records, files and working notes. Late writes from the affected turn cannot recreate notes. The original task objective remains available for rebuilding the work.

## Continuation and accounting

Provider requests share `ModelRequestQueue`. `runtime.modelConcurrency` defaults to four simultaneous calls, with one slot per authenticated actor. Waiting foreground calls precede background jobs. Queued cancellation removes the call without provider dispatch or a usage record. This is concurrency control, not a task call, time or cost budget. Capacity changes apply as active calls drain.

Approval requests and decisions live in `task_approvals`, including exact batch items, corrections, timeouts and the deciding identity. Requests commit before displaying controls; decisions commit before execution proceeds. Restart invalidates pending prompts. Historical approvals are audit records, never reusable grants. Steering commits as ordered task events before `/steer` acknowledges it. A failed save pauses execution.

Owners can explicitly resume a paused, failed or cancelled task with `/talk message:... task_id:...` in its current channel. To move it to another enabled location, use `handoff:true ephemeral:true`. Handoff checks original ownership and current source access, moves the task atomically, and requires private responses thereafter. A second concurrent claim fails. The runtime restores the objective, notes, goals, evidence and steering, then uses current access policy and fresh approval gates. Completed tasks and tasks with pending approvals or unknown action outcomes cannot resume. Startup never automatically replays tasks.

Startup now acquires an OS-owned instance listener before login or store recovery. A second process using the same canonical storage root is rejected. Windows uses a named pipe; other platforms use a deterministic loopback port and fail closed on an address collision. The OS releases ownership when the process exits. Distributed workers remain outside this deployment model.

Every provider attempt records its model, purpose, status, reported tokens, duration and optional configured-price estimate in `model_usage`. Async task ownership follows compaction, transcription, skill evaluation and other nested model work. Dreaming records its originating task and job. Failed calls retain an error code with unknown tokens and cost where the provider returned none. Old `task_usage` records remain readable. Estimates exclude unreported charges and cache discounts; they are not billing receipts. `/tasks task_id` exports these records privately across restarts. There is no cost-based stopping rule.

Each model request estimates message, tool-schema and image overhead, reserves output space and a safety margin, and prunes or compacts before dispatch when necessary. A context that still cannot fit pauses with task state intact. Provider tokenizers and image accounting differ, so this is a capacity estimate. Compaction keeps complete tool-call/result groups and the original request; summaries and scratchpad excerpts never become system instructions.

An owner can resolve an unknown action using `/tasks task_id action_id resolution verification` after checking its actual effect. The ledger labels this an owner attestation. `/tasks verify:true task_id action_id` and `verify_action` can instead inspect supported Discord postconditions. Definite absence verifies deletions; exact bot message content verifies supported edits or sends with a recorded message ID. Permission errors, partial batch deletion and unsupported operations remain unknown. Verification never retries the action or resumes the task. Once every uncertain action is resolved, explicit resume becomes available.

Prior answers carry their source references into later turns. Untracked legacy summaries are excluded from model context. Source permissions are checked before and after model calls. Deletion interrupts active readers; the execution performing that authorized deletion may still confirm its own action. A continuation discards working notes whose sources are no longer available and rebuilds from the original objective and current evidence.

`/steer task_id` selects one exact active task when several are running. A text reply to the original request or its progress message steers that task while it accepts corrections. The authenticated actor and channel must match; another user's reply cannot redirect the work. The correction is persisted before acknowledgement. Once work has finished, replies use the ordinary conversation path.

## Deleting saved work

`/tasks forget:true task_id:...` and the destructive `task_forget` tool delete owned inactive work in its current location. Unknown actions and pending approvals must be resolved first. Deletion removes working notes, files, collections, evidence, action history and usage, plus operational request traces and pending dream-derived skill drafts. Minimal deletion markers prevent late learning jobs from recreating those drafts and allow cleanup retries. Original Discord messages, separately remembered facts, approved skills and operator migration backups are separate records and remain. Use their own deletion controls where needed.

## Named collaborators

The owner can use `/tasks task_id:... collaborator:@person` to permit an already authorized user to steer a public guild task, and `remove_collaborator:true` to revoke that permission. The named user uses `/steer task_id:... instruction:...` in the task channel. Membership is durable; access, active status, channel and membership are checked again when accepting and saving the correction. Private tasks and private handoffs do not accept collaborators.

Corrections carry the authenticated contributor ID. They force owner approval for subsequent changes, including after resume; contributors never acquire the owner's Auto-approve authority. Task inspection, file exports, action resolution, cancellation, handoff and approvals remain owner-only. The owner's task export lists collaborators. Forgetting the task removes its collaboration grants.

## Ordinary conversation controls

Users can describe earlier work without copying IDs. `task_search` finds eligible owned objectives in the current location; `task_control` steers, cancels or selects an inactive task for continuation. The normal model loop interprets intent and clarifies ambiguous descriptions. It does not classify every reply as a correction. A continuation finishes the selecting turn first, then reopens the selected workspace through the same checks as an explicit resume. Completed tasks may reopen for requested revisions. Unknown action outcomes still block reopening.

Task control is an internal execution action, classified with scratchpad operations rather than external Discord writes. It cannot grant access or approve a mutation. Private tasks stay private; cross-location transfer still uses the explicit private handoff flow.

Cancellation propagates to active chat-completions/Responses HTTP requests and single/batch approval waits. Approval signals are runtime-owned, separate from serialized requests. Cancelling removes pending controls and settles the approval record without dispatch; an already submitted external action still needs its receipt or verification.

## Evidence changes during a request

Outdated channel context and prior turns are excluded before prompting. Edits queue an in-place evidence refresh without cancelling the execution. DiscordHistoryReader re-fetches each affected message; Runtime replaces stale context and retains valid tool results and action receipts. A new evidence record separates refreshed work from obsolete snapshots. Deleted or unreadable messages become specific gaps; other work continues. Unstarted mutations wait for pending evidence changes to be processed, then use normal authorization and approvals. User cancellation and tool-call accounting remain intact. Transient bot task controls are excluded from ingestion and history-page results. Partial Discord updates are fetched and compared before being treated as edits.

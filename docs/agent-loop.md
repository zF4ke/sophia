# Agent loop

Sophia uses native function calling in one model-led loop. It does not use a separate deterministic intent router.

## A turn

1. Normalize the authenticated event, including attachments, reply authorship and response visibility.
2. Create or verify an owned task in the current location. Explicit resume claims inactive work atomically and refuses unresolved actions.
3. Load recent conversation context and the current task's saved evidence, notes, goals and eligible memory hints.
4. Render `runtime/agent_loop` from authenticated identifiers and configuration. Put source context in separate messages, then retain the actual user request.
5. Estimate the next request including tool schemas and media overhead. Reserve output space, prune or compact if necessary, and pause if the required context still cannot fit.
6. Send the selected model the transcript and visible tools. Persist returned main-loop usage before applying its decision.
7. Recheck steering and authority. Validate calls; obtain caller-owned approval for changes when required; persist mutation intent before dispatch.
8. Save tool evidence and action receipts. Append every tool response before inserting visual evidence for the next model turn.
9. Accept a supported final answer or continue useful tool work. Persist outcomes and bind actual delivered messages through the adapter.

## Task continuity

`ExecutionControl` is shared across continuation turns. There is no default total call or time limit, and no note quota. An explicit `runtime.toolCallLimit` pauses at its configured threshold; zero disables it. Tool/network/container deadlines bound individual operations and do not declare the task complete.

The same task owns all continuation notes, plans, goals, files and evidence. Continuation preserves the original user instruction and carries unfinished state as source context. It never claims new approval, invents user unavailability or starts a second executor. Paused and failed results do not automatically restart.

`/steer` persists corrections from the authenticated owner or a named collaborator on a public task. Collaborator corrections carry their user ID and require owner approval for subsequent changes. A correction invalidates a stale model decision and skips unstarted calls; completed effects remain recorded. `/stop` cooperatively cancels the owner's work. In-flight external operations can finish after cancellation.

## Narrow failure handling

Capability/schema validation happens before dispatch. Repeated unchanged calls and repeated failures trigger bounded corrections or pause. Useful changing cursors can continue. Provider errors, missing credentials and context capacity failures remain distinct from completion.

Mutation receipts record exact validated arguments, dispatch intent and returned resource identifiers. A lost connection or failed receipt after dispatch is an unknown outcome. The runtime pauses before asking the model to retry. Explicit owner verification can resolve uncertainty, but never executes the action again.

## Context and sources

`task_tool_runs` retains full results beyond model context trimming. `source_read` paginates a captured web source without refetching. Context selection is not a reusable tool-result cache and cannot substitute old observations for requested current reads.

Compaction keeps the original request and whole tool-call/result groups. The summarizer receives source excerpts as user data; summaries remain assistant working context. Voice changes presentation only. Prompt interpolation is single-pass.

## Delivery

Normal conversation uses text. Cards and files use their dedicated delivery paths. Private interaction approvals remain ephemeral. Scheduling requires a saved schedule receipt; an ordinary finish cannot promise future execution. Conditional scheduled checks may suppress an unchanged completed result while preserving its findings. Failure and blocked-tool outcomes remain reportable.

See [task lifecycle](task-lifecycle.md), [access policy](access-policy.md), [workspace and media](workspace-and-media.md), and [the remaining v5 release work](v5-plan.md).

## Conversational follow-ups

The normal model loop can use task_search and task_control to interpret mentions and replies as steering, stopping or continuation, while unrelated questions remain new turns. Task controls operate on owner-scoped work and never authorize external writes. The selection turn finishes before a prior workspace reopens; its sources and unresolved actions are rechecked.

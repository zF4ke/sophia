---
title: Work, steer, stop, resume
sidebar:
  order: 3
---

Sophia keeps a durable task for work that needs more than a single answer. It can save a plan, working notes, goals, evidence, files, usage and action receipts. You can inspect these through `/tasks` and the task's Details control.

## Describe the work, not its ID



> **You**: Compare the last three community events and suggest improvements.
>
> **Sophia**: I'm collecting the event discussions. [Stop] [Details]
>
> **You**: Sophia, focus on attendance. Ignore the discussion about prizes.
>
> **Sophia**: I'll compare attendance and the reasons people gave for missing each event.

Steering is applied at cooperative boundaries. A change cannot undo an external action already in flight. Say "Sophia, stop the event comparison" to cancel that work. `/stop` and Stop remain available.

## There is no default task budget

Tasks do not stop after an arbitrary number of calls, elapsed minutes or dollars. Context capacity, per-operation timeouts, concurrency and repeated unchanged failures still need handling. An operator can set an explicit tool-call limit; zero disables it. A pause is not a claim that the work is finished.

## Resume explicitly

After a process restart, interrupted tasks pause. Say "Sophia, continue the event comparison from yesterday." Sophia can find the owned task from its description and resume it with its saved workspace. If several tasks fit, she asks which one you mean. `/tasks` and `/talk task_id:...` remain available for precise selection. Sophia checks owner, status, location and session. Unknown external action outcomes must be verified before continuing. It does not replay a potentially successful write just because the network response was lost.

Private cross-location handoff requires the authenticated owner and an eligible task. Moving work does not remove its source restrictions. Saved task exports and files remain owner-controlled.

## Invite a collaborator

For a public guild task, the owner can use `/tasks task_id:... collaborator:@person`. The person needs their own Sophia grant. They can then use `/steer task_id:... instruction:...`. Add `remove_collaborator:true` to remove them.

Collaborators do not own your approvals, private files or cancellation controls. Their steering makes later changes require owner approval. Private tasks cannot add collaborators.

## Forget a task

Use the task's supported forget controls or ask Sophia to forget an owned task. Forgetting removes its retained record and associated usage from reports. It does not delete messages already sent to Discord or retract files someone downloaded. See [costs](../costs/) and [command reference](../../build/commands-and-admin/).


An ordinary mention or reply can be a correction, a stop request, a question about progress, or a new topic. Sophia uses the message's meaning and eligible task descriptions to decide. Saying 'stop discussing prizes' changes the scope; it does not cancel the whole analysis. Task discovery currently stays within your account and conversation location. Private handoff remains explicit.

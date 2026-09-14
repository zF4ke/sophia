---
title: One task, from question to result
description: A worked conversation connecting research, steering, files, approval, and reusable skills.
sidebar:
  order: 10
---

This simulated walkthrough uses an event review. Your account has access to the source channels and an Ask grant for changes. Docker is available for file analysis. The findings below are illustrative.

## 1. Ask for an outcome

> **You**: Compare our last three events. Read the planning threads and attendee feedback. Tell me what we should change next time.
>
> **Sophia**: I'll identify the events, collect the discussions, and separate recurring feedback from one-off complaints. [Stop] [Details]

Sophia can keep a plan and evidence while it searches. A thin search result is a starting point: it may need surrounding messages, threads, or indexing. Ask for coverage when an answer claims to represent a whole channel.

## 2. Redirect the work

> **You**: Sophia, focus on scheduling in the event review. Ignore prizes and game choices.
>
> **Sophia**: I'll compare attendance and scheduling conflicts. I'll leave the other feedback out of the recommendation.

Steering takes effect at a cooperative boundary. The task keeps its identity and gathered evidence. Inspect it through `/tasks`; you do not need to restart the request.

## 3. Bring in a file

> **You**: [attendance.csv] Use this for the counts. Flag duplicate registrations first.
>
> **Sophia**: Two entries share a registration ID. Should the latest registration replace the earlier one?
>
> **You**: Yes. Make a cleaned CSV and a chart too.
>
> **Sophia**: Here are the revised files. Counts come from your CSV; reasons for absence come from the discussions. [attendance-clean.csv] [attendance.png]

Mention Sophia or reply with follow-up instructions while work is active. Files and code belong to the task workspace. A chart does not establish causation: distinguish the counts from participants' explanations and Sophia's interpretation.

## 4. Review before publishing

> **You**: Draft a short recommendation for #event-planning. Include source links, but leave the attendance file private.
>
> **Sophia**: Draft: Try an earlier start time. Scheduling conflicts appeared in feedback for two of the three events. Check availability with a poll before fixing the date.
>
> **You**: Send that with the source links.
>
> **Sophia**: Send the recommendation to #event-planning? [Approve] [Deny]
>
> **You**: [Approve]
>
> **Sophia**: Sent to #event-planning. [Message link]

The real approval contains the concrete action and destination. It belongs to you and does not authorize a different message. Publication still requires eligible sources and Discord permissions. A lost response requires checking the action's outcome before retrying.

## 5. Keep the useful method

> **You**: Save this review method for next month: validate the CSV, read feedback, distinguish evidence from guesses, then draft for approval.
>
> **Sophia**: I'll save that method with its source history so we can reuse and revise it.

A skill saves the procedure. It does not pre-approve next month's publication. Dreaming, when enabled, can also derive useful lessons from delivered work without an explicit request.

## Pick up later

Inspect saved evidence, receipts, and files through task controls. After a restart, say "Sophia, continue the event review". She can find and reopen its saved workspace. Open `/costs` to see retained usage, including retries and missing estimates.

Read more about [research](../research/), [tasks](../tasks/), [files](../workspace/), [approvals](../permissions/), and [memory](../memory/).

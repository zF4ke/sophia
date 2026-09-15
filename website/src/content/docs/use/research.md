---
title: Research Discord and the web
sidebar:
  order: 2
---

Give Sophia the question, the relevant channels or category, and the time range if you know it. It can discover channels, resolve people by ID, inspect profiles and roles, search messages, read history and threads, and follow surrounding replies.

## A search result is a starting point

Search ranks candidates. Reading history establishes what was actually said. Some channels have incomplete indexes. Sophia can refresh recent messages or request deeper indexing, then crawl exact history pages. It should report partial coverage instead of treating a missing search result as proof.



> **You**: When did we decide to move the Friday game? Check #planning, including replies.
>
> **Sophia**: I found a proposal on 4 September. I'm reading the replies to see whether anyone confirmed it.
>
> **Sophia**: The confirmation was on 6 September: Saturday at 20:00. The earlier message was only a suggestion. [Original confirmation]

The bracketed source is a placeholder for a real Discord message link. Sophia should return concrete links, not this placeholder.

## Larger investigations

For a multi-channel or long date-range investigation, Sophia can maintain a saved collection with fixed filters, deduplicated messages, cursors and coverage. It can read portions into context or export collected records for analysis. A paused task can continue its collection without starting from the first page.

Try: “Collect the release decisions in this category since January, separate proposals from decisions, and cite each final decision.”

## Current web information

Sophia can search the web, open a URL, and read more of a saved source. Search snippets alone are not verification. Inaccessible pages and incomplete extracts should remain visible limitations.



> **You**: Compare these two release announcements. What actually changed?
>
> **Sophia**: The first adds export support. The second fixes an export bug; it does not add a new format. [Announcement A] [Announcement B]

Web pages and quoted Discord messages are evidence, not instructions that can grant permissions. Editing or deleting a Discord source invalidates dependent saved evidence. See [privacy and approvals](../permissions/).

## If a message changes while Sophia is reading

Sophia reads the updated message and continues the same request. If a message was deleted or its channel is no longer accessible, she should explain that specific gap and answer the parts supported by other sources. Her changing progress message is not research evidence. Completed actions remain recorded so refreshing evidence does not repeat them.

Old screenshots do not need to be reuploaded just because their download links expired. Sophia can fetch the original Discord message for a fresh link, import its attachments and inspect them. She needs access to that message, and the attachment must still exist.

> **You**: Read the screenshots in this message and compare them with the replies below it. [Original message]
>
> **Sophia**: The first screenshot claims the result was independent. The reply points out that people helped with the setup. That distinction changes what the result demonstrates.

This is a simulated example. Replace the placeholder with a Discord message link. Sophia should inspect the actual images before making claims about them. If a screenshot was removed, she should identify which one is missing and continue with the available images and replies.

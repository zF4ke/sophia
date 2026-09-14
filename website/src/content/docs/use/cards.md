---
title: Cards, polls and revisions
sidebar:
  order: 7
---

Use plain text for ordinary answers. Ask for a card when pages, navigation, buttons or a structured presentation make the result easier to use.



> **You**: Make a two-page guide for the event: schedule first, rules second.
>
> **Sophia**: [Event guide: page 1 of 2] [Next]
>
> **You**: Change the starting time to 19:30 in the existing guide.
>
> **Sophia**: Updated the schedule in the original card. [Message link]

Cards retain state and revisions. Sophia can inspect an existing card before editing it. Concurrent clicks and edits are serialized. Creator identity and source permissions govern access. Expired card cleanup can retry after a temporary failure. Optional card scripts use the isolated container.

## Native Discord polls



> **You**: Poll: Saturday or Sunday for the next game?
>
> **Sophia**: Create a poll with “Saturday” and “Sunday”? [Approve] [Deny]
>
> **You**: [Approve]
>
> **Sophia**: [Native Discord poll]
>
> **You, later**: What's the result?
>
> **Sophia**: Saturday has 8 votes; Sunday has 5. The poll is still open.

The numbers above are simulated. Real results come from the poll's message ID. Polls and cards are separate features. Both need an eligible destination and the required Discord permissions. See [UI contracts](../../build/ui/).

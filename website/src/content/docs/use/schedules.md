---
title: Follow-ups and quiet monitoring
sidebar:
  order: 8
---

Ask Sophia to follow up once or on a recurring schedule. Supported timing includes intervals and timezone-aware daily or weekly recurrence. Use `/schedules` to inspect and manage your scheduled work.



> **You**: Every Monday at 09:00 Lisbon time, check the release notes. Tell me only if there is a new release.
>
> **Sophia**: I'll check on Mondays at 09:00 in Europe/Lisbon and stay quiet when the release is unchanged. [Schedule details]
>
> **You, later**: Move that check to Tuesday.

Creating or changing a schedule follows your permission policy. Each run checks current access again. A grant expiring does not leave a schedule with permanent authority. Conditional checks can finish without sending a message when there is nothing to report.

Schedules survive restarts. Uncertain delivery pauses rather than blindly sending the same result again. Cancellation stops future runs; it cannot retract a result already delivered. See [scheduling contracts](../../build/scheduling/).

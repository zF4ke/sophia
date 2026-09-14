# Scheduled work

`schedule_create` saves an explicitly requested future task for the authenticated owner and current destination. The immutable request in the task's action receipt includes its instructions and timing. The schedule approval covers its result delivery; actions the future task chooses still follow the current action tier and Ask/Auto mode.

Schedules support a one-time ISO timestamp, fixed intervals, daily local time or weekly local time. Every schedule carries an IANA timezone. Daily and weekly recurrence use [Temporal](https://github.com/js-temporal/temporal-polyfill) calendar arithmetic through daylight-saving changes. Ambiguous times choose the earlier occurrence, and missing local times advance through the gap. Missed recurring occurrences coalesce into one run rather than replaying every missed interval.

`Scheduler` checks for due work while foreground requests are idle. Each occurrence has a durable claim and uses the ordinary runtime. It rechecks the owner's current access and channel visibility before model work, at runtime authorization checkpoints, and before delivery. Cancellation or revision revokes an unstarted delivery. A send already in flight may still complete.

The schedule store records delivery intent before sending and records the Discord message ID afterwards. Restart pauses interrupted runs, including uncertain deliveries, without replaying them. Unused active schedules remain due after restart. Errors pause the schedule for review. `schedule_update` uses the current revision to replace the instructions and timing and create a future occurrence. `schedule_cancel` stops future work.

`/schedules` privately lists the current owner's schedules. Add `schedule_id` to export recent run records, linked task IDs and delivery outcomes. `schedule_list` exposes the same owned schedules to the model. Global availability and poll cadence live under `settings.scheduling`.

Use `notification_policy: "conditional"` for monitoring. The model receives the previous completed occurrence as source context and can set `finish.notify: false` after a successful check with no reportable change. Findings remain in the owned task record; the schedule advances without a Discord message. Missing notification decisions default to delivery. Paused tasks and runs with failed or blocked tools remain reportable. Ordinary conversations ignore quiet decisions.

Ephemeral guild requests currently cannot create or revise public delivery schedules. Use an enabled DM or an ordinary channel conversation. Tests cover ownership, revision conflicts, DST, coalesced intervals, quiet monitoring, cancellation and restart recovery. Live Discord delivery still requires validation.

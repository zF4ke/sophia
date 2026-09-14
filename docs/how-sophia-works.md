# How Sophia works

Sophia has one identity. Guilds, channels and tasks identify where work happens and who may see its sources; they do not create separate personas. The configured voice changes presentation, not identity or authority.

An authenticated Discord event enters the conversation adapter. Availability and the user's grants are checked before model work begins. The runtime binds a durable task, loads eligible context, and lets the selected model call tools until it answers or execution pauses. The model's arguments cannot grant permission or approve a change.

The task keeps its objective, notes, plan revisions, goals, steering, tool evidence, approvals, usage and action receipts. Continuation retains the original request. A restart pauses interrupted work; the user explicitly resumes it. An uncertain external action blocks resume until its outcome is checked. This prevents a lost response from becoming a duplicate action.

Research starts with what is actually available. Discord indexes may be partial. Retrieval returns concrete message IDs, links, coverage and a cursor that can be reused unchanged. Larger investigations can create a task-owned collection, collect deduplicated pages, read stable revisions and export bounded pages for analysis. Collection reads recheck source-channel access.

Web search and page reading use public destinations with DNS and redirect checks. Downloaded pages are source material. A saved source handle lets the model read further sections without repeatedly downloading or inserting an entire page into context.

The workspace is a Docker container with no network, host mounts or host credentials. Python, JavaScript and media utilities operate on task files. Image input requires an image-capable profile; video inspection produces frames with requested timestamps. Audio transcription requires a declared audio-capable chat-completions profile and extracts bounded audio windows in the container. Missing infrastructure produces an explicit error, not a host fallback.

Memory is stored in one local knowledge collection with ownership, scope, provenance, revisions and deletion markers. Idle dreaming considers delivered conversations and demonstrated tool use. It can propose useful memories and private procedural drafts without the user listing everything to remember. Draft procedures are not automatically shared or promoted. Source deletion invalidates derived memories and queued dreams; collection exports and their derived workspace files carry deletion provenance too.

Schedules reuse the conversation runtime and current authorization. Conditional checks include the previous completed result and can suppress an unchanged successful response. Paused work and errors still need attention. Missed or uncertain delivery does not authorize blind replay.

Context management bounds what enters a request, not how much useful work a task may accomplish. Compaction preserves the user's objective and complete tool-call/result groups. An optional explicit operator limit can stop calls across continuation turns; by default there is no total call or duration limit. Usage is recorded separately from execution permission.

See [the implementation plan](v5-plan.md) for acceptance criteria and unfinished release work. This description does not claim that live provider, Docker and Discord integration gates have all passed.

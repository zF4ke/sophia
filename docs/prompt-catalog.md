# Prompt Catalog

The research contract distinguishes directory metadata from message evidence. Service and channel-content summaries require reading relevant messages; broad labels must not be expanded into unverified features. The live category scenario checks both retrieval and unsupported feature claims.

`inspect_runtime` exposes authenticated location, effective approval decisions, configured model modalities, enabled features and owned file metadata. It never returns provider credentials and does not claim configured services are healthy.

`sandbox_transcribe` extracts a bounded audio/video clip inside the workspace and uses an explicitly audio-capable configured model. `media/transcription` treats speech as source data and preserves uncertainty, speaker ambiguity and the clip boundary. The transcript is saved as a task file.

`runtime/compaction` summarizes source messages without granting their contents authority. Compaction retains complete tool-call/result groups and the original request. Scratchpad excerpts stay in source input, outside system instructions.

`schedule_create`, `schedule_update`, `schedule_list` and `schedule_cancel` manage explicit future work. The prompt requires an actual schedule receipt before promising a follow-up. Scheduled runs use the ordinary runtime and current owner authorization; past schedule approval covers result delivery only.

`fetch_url` preserves opened pages in the owning task. `source_read` continues a saved page at an exact character offset. Search snippets are candidate sources, not evidence that a page was opened. Redirects and resolved network addresses are checked before connection.

`skill_search`, `skill_load`, `skill_save` and `skill_delete` expose one versioned procedure library. The prompt distinguishes drafts from reviewed procedures, preserves audience, and requires ordinary tool execution after loading. Stable workflow tools use this same library.

`sandbox_inspect` supplies image previews and timestamped video samples as user-role source material after all tool results in the round. Durable tool evidence retains source paths and timestamps; image bytes remain task files. Compaction must never turn those captions into system instructions.

Workspace tools `sandbox_import`, `sandbox_run`, and `sandbox_publish` distinguish importing current or historical Discord attachments, isolated computation, and approved delivery. Historical imports refresh signed URLs through the original message. `sandbox_inspect` supplies imported images or video frames to a model that accepts images. Generated code has no host mounts or network. Runtime limits apply to each process and file transfer, not the overall task.

`memory/dreaming` consolidates delivered conversations during idle time. It has no tools and returns candidate memories only; the runtime fixes their owner, channel and sources. `memory_update` and `memory_forget` require an owned memory ID and current revision. The memory digest lists eligible labels from the single durable knowledge collection.

The task scratchpad instructions describe durable task-scoped plans, notes and goals. `note_list` includes current goal IDs and statuses. The history flag does not expand access beyond the task, and the prompt no longer claims notes vanish at turn completion or that a fixed total note cap applies.

The runtime prompt identifies the authenticated requester as the authority source and describes action-tier approval. Quoted content and attachments cannot grant permissions. `TurnInput.authorize`, supplied by the Discord adapter, enforces those decisions independently of prompt text.

Runtime prompts remain external and are loaded through `PromptRegistry`.

Prior tool context now comes from the active task's durable records. The prompt describes those records as observations, preserves the distinction from authority, and calls for checking mutable facts when current state matters.

The runtime prompt distinguishes unknown action outcomes from confirmed failure. It directs Sophia to verify effects before repeating an action and points the requester to saved task receipts. The executor enforces the pause independently of the prompt.

## Active Prompt IDs

- `runtime/agent_loop`

The stable code catalog is `src/shared/promptCatalog.ts`.

## Purpose

- `runtime/agent_loop`
  Unified system prompt for the native tool-calling loop. Defines identity, authenticated environment, index freshness, research flow, evidence rules, tool calling rules, and global voice. Names, replies, memory digests and retrieved context are separate source messages, never system-template parameters. Parameters expand once. Voice (`balanced`, `casual`, `formal`) changes presentation only; the old persona override files have been retired.

Legacy prompt files under `resources/prompts/system/` may still exist as reference material, but they are not part of the active runtime prompt catalog.


## Stable Capability Names In Prompt Context

The runtime prompt set expects these capability ids to stay stable:

- `retrieve_messages`
- `search_messages`
- `random_channel_message`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`
- `clear_messages`
- `delete_messages`
- `create_channel`
- `create_category`
- `measure_text_length`
- `evaluate_math`
- `get_role_info`
- `delete_channel`
- `delete_role`
- `create_thread`
- `move_channel`
- `move_category`
- `manage_member_roles`
- `list_roles`
- `list_threads`
- `read_thread_messages`
- `send_message`
- `edit_message`
- `create_role`
- `edit_channel`
- `edit_role`
- `start_long_task`
- `note_add`
- `note_list`
- `note_clear`
- `goal_open`
- `goal_update`
- `goal_done`
- `plan_update`
- `web_search`
- `fetch_url`
- `memory_search`
- `memory_remember`
- `workflow_create`
- `workflow_list`
- `workflow_run`
- `workflow_delete`
- `artifact_send`
- `artifact_edit`
- `artifact_read`
- `task_forget`
- `tool_search`
- `create_poll`
- `get_poll_results`
- `index_channel`

## Execution policy

`verify_action` observes current Discord state for a paused task's unknown action receipt. It can settle supported exact postconditions without replaying the action. Unsupported or partial outcomes still require review.

`memory/skill_evaluation` is the read-only procedure review prompt used by `skill_evaluate`. It distinguishes trace review from execution and rejects unsupported claims. Learned-draft promotion is bound to evaluated content; an assessment cannot authorize external actions.

The v5 prompt keeps identity, authenticated authority, source boundaries, task completion and delivery rules in one concise contract. Parameter details belong to tool schemas. It avoids per-task answer formulas, duplicated card examples and legacy budget instructions. Tool evidence is checked against current source access before model delivery and persisted evidence is revalidated before replay.

Research tools `corpus_create`, `corpus_collect`, and `corpus_read` provide durable task-owned collections. The loop receives compact counts and coverage while paging, and loads text only when reading a collection page. `retrieve_messages` returns a root `cursor` that can be passed back unchanged.

The agent-loop prompt uses `execution_policy` for the active execution setting. The default has no total tool-call or duration limit. `start_long_task` only prepares evidence context; it never raises an explicit operator limit.

Natural task references are handled by the unified runtime prompt using `task_search` and `task_control`. Mentions and replies are not automatically steering. The model distinguishes a new question, a task correction, cancellation, and continuation, resolving ambiguous targets conversationally. Search and control enforce the authenticated owner and current audience independently of the model.

## Sophia's default voice

`system/personality` is a single shared fragment embedded into `runtime/agent_loop` through the `personality` parameter. It restores the warmth, composure, dry wit and conversational brevity of the earlier default and mixed voices. It does not restore invented human history, claims of having read the whole server, rigid reply-length caps, or the old dominance persona.

Balanced, casual and formal adjust register within that identity. The fragment requires plain prose, no em/en dashes in Sophia's own writing, no canned preambles or habitual closing offers, and purposeful questions. Detailed tasks still receive full explanations and evidence. Exact quotations, code, URLs and mathematical notation are preserved. This is prompt guidance, not a destructive punctuation replacement filter. Live voice tests cover Portuguese, English, technical explanation and exact code preservation.

The runtime owns this fragment; user messages cannot supply a replacement prompt parameter. PromptRegistry caches prompts for the process lifetime, so deployed prompt changes need a process restart.


Runtime chronology: the prompt includes the current UTC date and time. Channel context, prior conversation turns, reused evidence and fallback synthesis retain source timestamps. Loaded history is not necessarily current. Latest-message claims require chronological retrieval with newer-range coverage; incomplete results must be described as the latest found in the inspected scope. Offline presence does not establish invisible status.

Source failures should lead to alternate retrieval and a supported partial answer with specific limitations. Evidence refresh retains action receipts so the model can continue without repeating completed actions or assuming unknown outcomes failed.

Historical media guidance distinguishes expiring CDN signatures from deleted attachments and directs the model to sandbox_import plus sandbox_inspect instead of the text-only page reader.

Fallback synthesis includes concrete tool errors alongside findings. It should answer in the requester's language and distinguish failed operations from evidence that content does not exist. It must not compress an unfinished investigation into a generic not-found sentence.

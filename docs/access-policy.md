# Access policy

Sophia has one identity. Availability determines where she may operate; a user or role grant determines who may request work and which actions they may authorize. Guilds do not confer separate identities.

## Availability and configuration

`storage/settings.json` holds `guildAllowlist` and `access`. An empty guild list enables no servers. Direct messages are disabled unless `access.directMessages` is true. A global user grant still requires an enabled location. Message triggers support guild text channels, threads and explicitly enabled DMs.

Operators can set `/access enable_dms:true` and grant `/access user:@person global:true level:write mode:ask`, including from DMs. Enabling DMs does not grant access to everyone. `enable_here` and role grants require a guild context.

An existing operator can enable a server with `/access enable_here:true`. To authorize a person there, use `/access user:@person level:write mode:ask`. Use `role:@role` instead for a guild role, `global:true` for a user grant across enabled locations, or `remove:true` to remove that specific grant. `enable_here:false` disables the server. Grants default to read-only and Ask mode. These settings persist without restarting; registering the new slash-command options requires the normal command registration step.

The `/access` command and its administrative controls remain available to authenticated operators in disabled locations. This exception only configures access; it does not enable conversation or tools. Existing operator and moderator records remain in use during migration. Operators have full action tiers, with changes asking by default. Moderators have conversation/read admission, but need an explicit grant for changes. No role name, display name, message text, or model argument establishes an actor's identity.

## Grants and decisions

`/talk approval_mode:ask` makes a task more restrictive even when its grant allows Auto-approve. The preference persists through continuation, pause and handoff. `approval_mode:inherit` returns to the current grant policy; it cannot exceed that policy. Reads remain free. Task exports show the saved mode. Expired approval prompts are recorded as `expired`, not as a requester denial.

| Highest granted tier | Reads | Writes | Destructive actions |
| --- | --- | --- | --- |
| `read` | Allow | Deny | Deny |
| `write` | Allow | Ask or Auto-approve | Deny |
| `destructive` | Allow | Ask or Auto-approve | Ask or Auto-approve |

Matching grants combine: an applicable Auto-approve grant permits an action only within its own tier. User grants can be global or guild-scoped; role grants are guild-scoped. Role membership is fetched afresh when evaluated. A failed role lookup supplies no role authority. Operators ask for changes unless they have an applicable explicit Auto grant or Allow rule. The legacy `runtime.autoApproveWrites` switch is removed during settings migration; it cannot carry an implicit grant into v5. Protected destructive targets stay blocked even in Auto-approve mode.

Changing member roles and moving channels require the destructive tier because they can change resource access. Role changes validate all requested role IDs before applying any, and report IDs and mentions for actual changes. Protected-channel checks also run inside `ToolExecutor`, so alternate callers and changes to protection while queued cannot bypass them.

Slash commands, message triggers, autocomplete and component clicks pass admission before invoking their handlers. Legacy public-command flags cannot admit an otherwise unauthorized user. Message triggers retain their existing enablement and rate-limit settings. Enabled channels can ingest messages by other authors as retrieval sources; ingestion does not grant those authors permission to invoke Sophia.

The conversation adapter binds the actor to the authenticated Discord event. Runtime admission and every capability execution recheck current access. Revocation or disabling the location prevents subsequent calls; an already dispatched external action may finish. Ask-mode execution requires an approval supplied by the runtime for that invocation. The requester may decide their own prompt if their current grant still permits its tier. Operators may also decide prompts; unrelated ordinary users cannot. Single, batch, confirmation and correction paths apply this identity rule.

Artifact script sends are proposals. They stay inside the originating guild, show the exact content when approval is needed, and execute through `ToolExecutor` with current requester authority. Card navigation and local game state remain ordinary interactions. Continuation progress uses the originating turn's notifier; an ephemeral turn does not silently publish progress to the channel.

## Expiry and inspection

Mutations are serialized per guild, including work from simultaneous tasks. Reads and independent guilds continue concurrently. Source-derived messages, polls, cards and files check their actual publication destination before dispatch. A private model context does not confer permission to disclose restricted sources publicly.

`/access expires_at:2027-01-01T12:00:00Z user:...` sets an optional expiry. `expires_at:never` removes it; omitting the option preserves an existing expiry. Both user and role grants support expiry. Expired grants confer no authority, including during an existing task. Independent operator or moderator authority still applies. The private Autorizações panel lists principal IDs, guild scope, tier, approval mode and expiry, with pagination.

## Source audiences and continuity

Retrieval, saved evidence, derived files, memories and skills check current source access. Public answers can use restricted history only in its source channel. Private replies may use other readable channels and enabled guilds when the requester still has access. Deleted sources invalidate dependent content and interrupt active readers. Source checks run before and after model calls.

Private facts and private dream-derived skills stay private to their owner. Only whitelisted presentation preferences follow the user into public conversations. Guild-shared skills cannot contain private source material. One identity does not make all stored facts available to every audience.

An owner can explicitly move inactive unresolved work to another enabled location with `/talk task_id:... handoff:true ephemeral:true`. Source access is revalidated, the move is atomic, and the task remains private. Unknown actions and pending approvals block handoff and resume. Approval history survives restart but never authorizes a replay.

Resource access follows current Discord permissions and protected-resource settings. Grants are per user or guild role; arbitrary per-resource grant expressions are not supported. Internal callers without an `authorize` callback remain a trusted compatibility boundary. Discord adapters supply it. Background crawlers use location availability rather than a conversational actor.

## Capability and tier rules

Operators can use `/access user:... capability:send_message rule:allow` for an exception to Ask mode within an existing grant. Use `rule_tier:write` for a whole action tier, `rule:deny` to prohibit it, or `rule:remove` to remove the matching rule. Rules can target a user globally or in the current guild, or a guild role. They appear alongside grants in the paginated access panel.

A matching deny wins, including for operators; Ask wins over Allow when rules overlap. A rule cannot exceed the active grant ceiling or enable a disabled location. Permitted reads remain prompt-free. Role membership is refreshed, and an unavailable membership lookup cannot bypass a possible role deny. Runtime approval and final tool dispatch both evaluate the actual capability ID. Task-level Ask remains more restrictive. Per-message or arbitrary resource expressions are not part of this rule format.

Named task collaborators can steer public tasks in the same channel, but cannot inspect private task records or approve effects. Any collaborator steering forces owner approval for later changes. See `task-lifecycle.md`.

# Discord UI

Ordinary replies use ordinary text. Cards are for navigation, actions and content that benefits from persistent controls. Keep the user's language, use concrete wording, and avoid decorative dashboards or repeated status paragraphs.

## Progress and delivery

Message requests use native typing. Longer work has one coalesced progress message with task-bound Stop and Details controls. Rotating reactions are removed. `/talk` uses Discord's deferred response and respects the selected private response mode.

Private progress, edits and cleanup use ephemeral interaction webhooks. An expired private delivery path must never become a public fallback. `/tasks` posts eligible public-task summaries in the channel by default; `ephemeral:true` includes private summaries for the requester. Task inspection and exports also default to the channel; use `ephemeral:true` for a private reply.

Stop acts only on the authenticated owner's task. Details shows that task's current state. Steering arrives through `/steer` and is persisted before acknowledgement.

## Approvals

Reads within a grant run without prompting. Ask mode shows concrete changes and their targets before execution. Auto-approve applies only within the authenticated user's granted action tier. Protected targets remain unavailable.

Approval cards support accept, decline, correction and stop. Batches list their actual items and may support category selection. Changed arguments require a new decision. A timeout means that no decision arrived; it is distinct from a user declining. Destructive Ask-mode actions retain their confirmation step.

Only the requesting user can approve. Approval controls commit a decision to the task ledger before execution. Restarted or expired prompts cannot grant authority.

## Persistent cards

`artifact_send` creates a card. `artifact_edit` changes the stored card in place. Specs, view state, game state, creator IDs and Discord message IDs persist in `products.sqlite`. Controls are routed centrally and continue to work after restart.

The owner or an operator can change a card's structure. Cards without recorded ownership require an operator. Every click still checks current access. Per-card serialization prevents concurrent clicks and structural edits from overwriting each other.
`artifact_read` exposes the current revision and exact historical revisions to the owner or an operator. `artifact_edit expected_revision` rejects an edit when the card changed since inspection. Specs and handler state are snapshotted together after structural changes and script interactions. Navigation alone does not create another content revision.

Navigation supports section selection and pagination. Specs may include galleries, files, link buttons and supported Discord selects. Validation enforces the current component and text limits before sending. Native polls use Discord's poll API.

Script handlers execute in the Docker workspace. The JavaScript VM inside the container is a language runtime, not the host isolation boundary. External actions proposed by a handler use ordinary authorization and approvals.

Cards persist by default. A configured expiry removes the message; failed cleanup retains its record for a later retry. An unavailable card or handler returns a private error to the clicker.

Cards retain source links across revisions. Reading, editing and interacting with a card recheck those sources; a deleted or inaccessible source blocks redisplay. This does not retract already published Discord messages. Successful expiry cleanup removes the card's stored revisions, ownership and source metadata. If a card was sent but saving its state fails, the result preserves its message ID and reports the failure; the runtime pauses without sending a duplicate.

## Settings and inspection

`/settings` opens in the channel by default; `ephemeral:true` opens it privately. Operator checks still protect every control. The installation-wide costs view updates the same settings message. It separates model selection, voice, service switches, compaction and advanced runtime controls. Voice styles are balanced, casual and formal. Service switches pause memory consolidation, workspace execution or scheduling without deleting their data.

Category resets preserve unrelated settings, access rules and storage paths. The default task call limit is zero, which disables the cap. Availability and approval grants are configured through `/access`, including explicit DM availability.

`/skills` inspects owned skills. Operators can review quarantine entries and adopt one exact revision as a private draft. `/tasks` exports owned state, source-checked evidence, files and usage records. Diagnostic exports may contain stable IDs; ordinary confirmations use Discord mentions and links when available.

## Acceptance

`/memories` searches eligible facts. Operators can page through `quarantine:true` and adopt an exact legacy revision as private memory. `/access` includes a paginated grant list with principal IDs, scope, tier, mode and optional expiry. `/tasks forget:true task_id` deletes inactive resolved work; separate memories and published artifacts have their own controls.

Check ordinary conversation, private progress, approval ownership, expired controls, concurrent card clicks and restart recovery. Verify wrapping, labels, navigation and tap targets on real desktop and mobile Discord clients before release. Synthetic component tests do not establish mobile usability.

Personality comes from the shared `system/personality` prompt. Balanced restores Sophia's relaxed default, casual is more informal and formal is more reserved. All modes use the same plain-writing rules and preserve exact source quotations and code.

Settings navigation has one button per destination. Serialized custom IDs must be unique across the whole message, including nested containers; settingsFeatures.test.ts checks every tab.

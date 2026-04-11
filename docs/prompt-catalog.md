# Prompt Catalog

Runtime prompts remain external and are loaded through `PromptRegistry`.

## Active Prompt IDs

- `system/base`
- `system/personality`
- `runtime/plan_turn`
- `runtime/select_next_step`
- `runtime/judge_evidence`
- `runtime/synthesize_answer`
- `runtime/debug_summary`
- `guards/insufficient_evidence`

The stable code catalog is `src/shared/promptCatalog.ts`.

## Purpose

- `system/personality`
  Defines Sophia's conversational voice, tone rules, and anti-patterns for natural speech. Loaded alongside `system/base` only for the synthesis call.
- `runtime/plan_turn`
  Chooses conversation vs Discord retrieval, taking active resolved targets and any active retrieval session into account.
- `runtime/select_next_step`
  Chooses the next single capability for the bounded retrieval loop, including retrieval continuation when an active scoped read already exists.
- `runtime/judge_evidence`
  Decides whether the current evidence bundle is enough to stop or continue, with explicit history-vs-semantic evidence tradeoffs.
- `runtime/synthesize_answer`
  Produces the final user-facing answer from the current mode, confidence, laneed evidence, and conversational recovery rules.
- `runtime/debug_summary`
  Reserved for optional graph-aware debug summarization.

## JSON Contracts

These prompts return strict JSON and must stay aligned with the runtime contracts in `src/runtime/contracts.ts`:

- `runtime/plan_turn`
- `runtime/select_next_step`
- `runtime/judge_evidence`

If you change a prompt’s JSON contract, update:

- the prompt file
- `src/runtime/contracts.ts`
- `src/runtime/Runtime.ts`
- this document
- the related tests

## Stable Capability Names In Prompt Context

The runtime prompt set expects these capability ids to stay stable:

- `retrieve_messages`
- `resolve_member_identity`
- `list_guild_structure`
- `resolve_channel_targets`
- `get_member_profile`
- `list_members`
- `get_guild_context`

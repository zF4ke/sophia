# Reusable skills

Idle dreaming can draft one procedure from successful tool evidence in a delivered task. The runtime accepts only demonstrated capability names and fixes its owner, source location and provenance. Draft insertion is idempotent by dream ID; matching names, including retired ones, prevent repeated recreation. Drafts are private to the owner and source location. `skill_save` reviews or revises the draft through ordinary approval; dreaming never promotes or publishes it. Deleted source links make derived skills unavailable, including after revisions.

`SkillStore` owns one library in `products.sqlite`. A skill contains a description, instructions, required capability IDs, examples, status, owner and audience. Every edit requires the current revision and writes an immutable version record. Retirement leaves a tombstone so migration cannot recreate a deleted legacy workflow.

`skill_search` returns descriptions for discovery. `skill_load` supplies full instructions only when useful. `skill_save` creates or revises an owned skill; `skill_delete` retires it. Saves follow write approval, and retirement follows destructive approval. Loading a skill never executes its instructions or approves its steps.

New skills default to the source channel and owner. A guild-scoped ready skill is shared within its guild. Drafts remain visible only to their owner in the source channel, including drafts intended for eventual guild sharing. Revisions cannot silently broaden audience. Missing legacy ownership or audience remains conservative; ambiguous DM procedures are quarantined.

The stable workflow tools use this library for structured tool steps. `workflow_run` returns those steps for the normal runtime loop. Legacy workflow IDs and guild audiences survive migration. There is no nested executor and no separate workflow budget.

`skill_evaluate` checks that required capabilities were demonstrated in a completed owned task, then uses the configured model to review the method against accessible trace summaries. It stores a content hash, verdict and findings. Learned drafts require a passing assessment for matching content before promotion; changing the method requires another assessment. This is a trace review, not an independent execution test. Evaluation does not approve external actions.

`/skills` privately lists eligible procedures and exports details by ID. Operators can inspect `quarantine:true` and adopt a reviewed `id` with its exact `revision` and `adopt:true`. Adoption narrows the record to an operator-owned channel draft; it does not publish or execute it. Records from another guild cannot be adopted here.

Tests cover audience boundaries, draft visibility, revision conflicts, historical versions, retirement, source invalidation, evidence-bound evaluation, and operator-only quarantine adoption.

# Sophia v5 — Improvement Plan

## Diagnosis

Sophia's runtime is over-orchestrated. We built a multi-node LangGraph state machine with 4-5 separate LLM calls per research pass (`plan_turn` → `select_next_step` → `judge_evidence` → loop → `synthesize_answer`), each requiring its own prompt, contract, and parsing logic. Every major open-source agent harness (Cline, Kilocode, OpenCode) proves that a simple while-loop with native tool calling works better: faster, cheaper, more coherent, and dramatically simpler.

The core insight: **stop orchestrating the model and let it drive**. The model should see all context and decide — in one streaming response — what tools to call, when to stop, and what to say. Our job is to give it good tools, good context, and guardrails.

---

## Phase 1: Drop LangGraph, Adopt a Simple Loop

**Goal**: Replace the StateGraph with a single `runLoop()` function.

### What changes:
- Delete `StateGraph` construction in `Runtime.ts`.
- Replace with:
  ```
  while (not done) {
    1. Build messages array (system prompt + conversation history + tool results)
    2. Call model with native tool calling enabled
    3. If model called tools → execute them → append results → continue
    4. If model produced text only → that's the answer → break
    5. Check budget limits (max iterations, max latency) → break if exceeded
  }
  ```
- The "nodes" (`ingest_turn`, `load_context`, `load_checkpoint`, `load_memory`, `plan_turn`, `route_mode`, `run_research_loop`, `synthesize_answer`, `persist_result`) become sequential steps *before* and *after* the loop, not graph nodes.

### What we keep:
- `load_memory` logic (load recent turns, channel context, prior evidence)
- Evidence extraction from tool results (per-tool strategies stay)
- Tool capability registry (tools still registered the same way)
- Repeated-call protection
- Budget limits
- Conversation identity resolution
- Checkpoint persistence

### What we kill:
- `plan_turn` as a separate LLM call → fold into system prompt
- `select_next_step` as a separate LLM call → model picks tools natively
- `judge_evidence` as a separate LLM call → model self-judges
- `synthesize_answer` as a separate LLM call → model's final text *is* the answer
- Mode classification (`conversation` vs `research` vs `refusal`) → model decides
- `candidateCapabilities` → model sees all tools, picks what it needs
- Argument enrichment layer → model provides complete arguments

### Estimated impact:
- **LLM calls per turn**: 4-5 → 1-3 (one per loop iteration, most turns will be 1-2 iterations)
- **Latency**: 3-5x faster for research turns
- **Cost**: ~60% token reduction (no duplicate context across separate calls)
- **Code**: ~50% reduction in `Runtime.ts` + `planning.ts`
- **Prompt files**: 4 runtime prompts → 1 system prompt (with tool descriptions)

---

## Phase 2: Native Tool Calling

**Goal**: Use the model's native function-calling API instead of structured JSON parsing.

### What changes:
- Define each Discord capability as a tool schema (JSON Schema or Zod → JSON Schema).
- Pass tools to the model API call.
- Model returns `tool_call` objects natively.
- Parse tool calls from the API response, not from free text.
- Feed results back as `tool_result` messages.

### Benefits:
- No more prompt-engineered JSON extraction.
- Model is trained on tool calling — better argument quality.
- Parallel tool calling possible (model can request multiple tools at once).
- Eliminates `select_next_step` prompt entirely.

### Discord tools to expose:
Same 7 (+1 meta):
1. `retrieve_messages` — search Discord message history
2. `resolve_member_identity` — find member by name/mention
3. `resolve_channel_targets` — find channels by name/mention
4. `get_member_profile` — rich member info
5. `list_guild_structure` — channels and categories
6. `list_members` — paginated member list
7. `get_guild_context` — guild metadata
8. `finish` — explicit completion signal (like Cline's `attempt_completion`)

---

## Phase 3: Unified System Prompt

**Goal**: One system prompt that gives the model everything it needs.

### Composition:
1. **Identity & personality** (from `system/base.md` + `system/personality.md`)
2. **Environment context**: guild name (ID TOO), channel name (ID TOO), requester display name, date
3. **Conversation history**: recent turns summary
4. **Channel context**: recent messages in the current channel
5. **Prior evidence**: carried-over evidence from previous turns (if continuation)
6. **Available tools**: described inline with parameters
7. **Behavioral guidelines**:
   - Search local cache first, escalate to live Discord if cache is weak
   - Don't guess — if evidence is insufficient, say so conversationally
   - When answering questions about Discord activity, always cite evidence
   - For casual conversation, respond naturally without tool use
   - Use `finish` tool when done (includes the final answer text)

### What the prompt replaces:
- `plan_turn.md` → behavioral guidelines handle mode routing
- `select_next_step.md` → tool descriptions handle selection
- `judge_evidence.md` → guidelines handle self-judgment
- `synthesize_answer.md` → model's final response handles synthesis

---

## Phase 4: Context Management

**Goal**: Handle long conversations without evidence loss.

### Strategy (adopt from Kilocode/OpenCode):
1. **Token counting**: track input/output tokens per model call.
2. **Overflow detection**: if approaching model's context limit, trigger compaction.
3. **Prune-then-summarize**:
   - First: erase verbose tool output text from old iterations (keep tool call metadata).
   - If still over: ask the model to summarize old conversation turns into a compact recap.
   - Insert summary as a system message, drop the old messages.
4. **Evidence pruning**: old evidence items from prior turns get summarized, not dropped.

### What we add:
- `isContextOverflow(tokenCount, modelLimit)` utility
- `pruneOldToolOutputs(messages)` function
- `compactHistory(messages)` model call (separate cheap model)
- Per-model context limits in `model-profiles.json` (already have some of this)

---

## Phase 5: Better Error Handling & Retries

**Goal**: Recover gracefully from model and tool failures.

### What changes:
- Classify errors: `retryable` (rate limit, 5xx, overloaded) vs `terminal` (auth, context overflow)
- Exponential backoff: 2s → 4s → 8s, respect `retry-after` headers
- Context overflow → auto-compact and retry (not crash)
- Tool execution errors → feed error message back to model, let it adapt
- Empty model output → retry once, then fallback answer

---

## Phase 6: Polish & Guardrails

**Goal**: Keep Sophia's unique strengths while using the new architecture.

### Carry forward:
- Reply-chain conversation identity
- Scope reset on goal shift (when topic changes, clear stale evidence)
- Repeated-call signature dedup (improved: warn model instead of hard-block)
- Budget limits (max iterations, max latency)
- Cache-first retrieval with live escalation
- Debug session tracing
- Evidence carry-over across turns

### New guardrails:
- **Doom-loop detection**: 3x same call → inject warning message, don't hard-block
- **Max iterations**: configurable per turn (default: 10)
- **Tool output truncation**: cap at 50KB per tool result
- **Finish tool required**: model must call `finish` to produce final answer (prevents runaway text)

---

## Migration Path

### Step 1: Proof of concept
Build the new loop alongside the existing graph. Feature-flag it. Test with live model calls.

### Step 2: Parity testing
Run both paths on the same inputs. Compare: latency, cost, answer quality, tool usage patterns.

### Step 3: Cutover
Once the new loop matches or exceeds the graph on all metrics, delete the graph path.

### Step 4: Cleanup
- Delete `plan_turn.md`, `select_next_step.md`, `judge_evidence.md` prompts
- Simplify `contracts.ts` (remove PlanDecision, StepDecision, EvidenceDecision types)
- Simplify `planning.ts` (remove `planWithModel`, `planNextStep`, `judgeEvidence`)
- Update all docs

---

## What We Don't Change

- **Discord integration**: commands, events, retrieval pipeline — all stay.
- **Tool strategies**: per-tool evidence extraction, argument validation — stays.
- **Storage**: libSQL, runtime traces, message cache — stays.
- **Security**: admin directory, rate limits, command policies — stays.
- **Prompt registry**: external prompt files — stays (but fewer of them).
- **Model gateway**: OpenRouter abstraction — stays.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Model doesn't self-judge well | Keep evidence quality guidelines in system prompt; if answers degrade, add a lightweight post-check |
| Model calls tools excessively | Budget limits + doom-loop + finish-tool requirement |
| Loss of fine-grained tracing | Trace every tool call and model response, same as before |
| Regression in conversation quality | Parity testing before cutover |
| Model doesn't understand Discord tools | Well-written tool descriptions with examples |

---

## Summary

The plan is: **simplify the orchestration, trust the model more, keep the Discord-specific retrieval logic intact.**

From 8 graph nodes and 4 runtime LLM calls → 1 loop and 1 LLM call per iteration.
From 4 prompt files with 4 contract types → 1 system prompt.
From ~1500 lines of orchestration → ~300 lines.

The Discord tools, evidence pipeline, conversation identity, and retrieval strategies are Sophia's real value. The over-orchestrated planning layer is not.

## Extra:

Redo all live tests with the new loop architecture, test the abilities in live_tests_raw.md, but add proper live tests later and update that markdown to be clean.
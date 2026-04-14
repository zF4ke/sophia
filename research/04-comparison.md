# Cross-Framework Comparison

## The Universal Pattern

All three frameworks (Cline, Kilocode, OpenCode) converge on the same fundamental architecture:

```
while (not done) {
  stream LLM response
  parse tool calls from response
  execute tools
  feed results back as messages
  check exit conditions
}
```

**No framework uses a DAG/graph orchestrator for the core loop.** Cline uses a raw while-loop. Kilocode/OpenCode use a while-loop wrapped in Effect.ts streams. None of them use LangGraph or similar.

## What Sophia Does Differently (And Shouldn't)

| Aspect | Cline/Kilo/OC | Sophia | Verdict |
|--------|---------------|--------|---------|
| **Orchestration** | Simple while-loop | LangGraph StateGraph with 8+ nodes | Over-engineered |
| **Planning** | Model decides inline | Separate `plan_turn` LLM call + classification | Over-engineered |
| **Tool selection** | Model picks tools directly | `select_next_step` LLM call after `plan_turn` | Over-engineered |
| **Evidence judgment** | Model self-judges (implicit) | Separate `judge_evidence` LLM call per pass | Over-engineered |
| **Mode routing** | No modes — model adapts | conversation/research/refusal classification | Over-engineered |
| **Answer synthesis** | Part of the main loop response | Separate `synthesize_answer` LLM call | Wasteful |
| **Tool args** | Model provides all args directly | `enrichArguments()` layer modifies model args | Unnecessary |

### The Core Problem

Sophia makes **4-5 separate LLM calls per research pass**:
1. `plan_turn` — classify intent, pick mode, suggest capabilities
2. `select_next_step` — pick next tool + arguments
3. `judge_evidence` — evaluate if evidence is sufficient
4. (repeat 2-3 per pass)
5. `synthesize_answer` — generate final response

Cline/Kilo/OC make **1 LLM call per loop iteration**. The model sees the full context and decides everything: what tool to call, when evidence is enough, and what to say.

### Why This Matters
- **Latency**: Each LLM call adds 1-5s. Sophia's multi-call approach is 3-5x slower than necessary.
- **Cost**: Each LLM call costs tokens. Sophia pays for ~5 calls where others pay for ~2.
- **Coherence**: The planner, step selector, judge, and synthesizer don't share reasoning. The model can't course-correct mid-thought.
- **Complexity**: 4 prompt files, 4 contract types, 4 sets of parsing logic, 4 potential failure points.

## What Sophia Does Right

| Aspect | How Sophia Does It | Others Don't Have This |
|--------|-------------------|----------------------|
| **Discord-specific retrieval** | Cache-first with live escalation | N/A — they don't need retrieval |
| **Conversation continuity** | Reply-chain identity, checkpoint restore | Others start fresh each time |
| **Evidence accumulation** | Cross-turn evidence carry-over | Others don't need this |
| **Scope reset on goal shift** | Clear stale context on topic change | Smart for multi-turn |
| **Repeated-call protection** | Signature-based dedup | All three have similar (doom-loop) |

## Key Patterns Sophia Should Adopt

### 1. Single-Loop, Model-Led Architecture
Let the model stream tool calls and text in one response. Kill `plan_turn`, `select_next_step`, and `judge_evidence` as separate LLM calls. Fold their concerns into the system prompt.

### 2. Native Tool Calling
Use the model's native tool-calling API (function calling) instead of parsing structured JSON from free text. All three frameworks do this.

### 3. Context Compaction
When context grows too large, prune old tool output text (keep metadata) and summarize if needed. Sophia currently just slices evidence to a fixed window.

### 4. Doom-Loop Detection
If the same tool is called 3x with identical arguments, intervene. Sophia has `maxRepeatedCallSignature` but it's a hard block — Kilo/OC prompt the model to try something different.

### 5. Model-Specific System Prompts
Different models need different instruction styles. Sophia uses one prompt for all models via OpenRouter.

### 6. Tool Output Truncation
Cap tool output at a sensible limit (e.g., 2000 lines / 50KB). Save full output to disk if needed. Sophia currently passes full tool output through.

### 7. Retry Policy
Classify errors as retryable vs terminal. Exponential backoff with `retry-after` header respect. Sophia currently has minimal retry logic.

# OpenCode — Architecture Study

## What It Is
CLI-based coding agent. Kilocode's upstream. Same Effect.ts foundation, same Vercel AI SDK, nearly identical architecture but cleaner (less vendor-specific baggage).

## Agent Loop
- `runLoop` function in `session/prompt.ts` — a plain **while loop**.
- Each iteration:
  1. Check exit conditions (finish signal, no pending tools, structured output).
  2. Load agent config.
  3. Insert context reminders (plan mode, system context).
  4. Stream LLM request: system prompts + history + tools.
  5. Process stream events via `SessionProcessor`.
  6. Execute authorized tools.
  7. Check context overflow → trigger compaction if needed.
  8. Decide: break or continue.
- `step` counter enforces max iterations.
- `ProcessorContext` tracks: toolcalls map, shouldBreak, blocked, needsCompaction, reasoning map.

**Key takeaway**: Identical to Kilocode at the core — while loop, model-driven, processor handles events. The simplicity is the point.

## Tool System
- Essentially identical to Kilocode. ~17 built-in tools + custom tools + MCP plugin tools.
- Same Zod schema validation. Same execute interface. Same truncation (2000 lines / 50KB).
- Same doom-loop detection (3x same call).
- **Tool resolution** filters by agent permissions before presenting to model.
- **Provider-executed tools**: some providers (DWS Agent Platform) execute tools server-side.

**Tools available**:
bash, read, write, edit, glob, grep, task, fetch, websearch, codesearch, skill, patch, LSP, question, todo, plan_exit

**Key takeaway**: The tool set is richer than Sophia's because it's general-purpose. But the dispatch pattern is identical: model picks tool → validate → execute → result back.

## Planning
- Same multi-agent pattern as Kilocode:
  - `build` (default) — full access
  - `plan` — read-only plan generation with 5 phases: explore → design → review → write plan → exit
  - `general` — multi-step subagent
  - `explore` — read-only fast search
  - `compaction`, `title`, `summary` — hidden internal agents
- **No pre-planning pass**. Model decides everything.

**Key takeaway**: Planning is just "a different agent with different permissions and system prompt". Not a separate orchestration framework.

## Context Management
- Token overflow detection: `isOverflow()` checks `input + output + cache.read + cache.write` against `model.limit.input - reserved`.
- **Compaction**: two-phase (prune tool outputs → summarize if still over).
- **Truncation**: tool outputs > 2000 lines or 50KB → save full to disk, return preview + path hint.
- **COMPACTION_BUFFER**: 20k tokens reserved for output.

**Key takeaway**: Same as Kilocode. The pattern is: detect overflow → prune old tool outputs → summarize if needed.

## Error Handling
- Same retry policy as Kilocode: exponential backoff, `retry-after` header respect.
- Same error categories: APIError, ContextOverflowError, etc.
- `retryable()` function classifies errors as retry-worthy or terminal.

## System Prompts
- **4-layer composition**: env (model-specific) → skills → instructions → user system.
- **Model-specific prompts**: `PROMPT_BEAST`, `PROMPT_GPT`, `PROMPT_CODEX`, `PROMPT_ANTHROPIC`, `PROMPT_GEMINI`, `PROMPT_DEFAULT`.
- **Environment info** injected: model name, working directory, git status, platform, date.
- **Structured output** gets a special system prompt instructing the model to use a `StructuredOutput` tool.

**Key takeaway**: The structured-output-as-tool pattern is clever — instead of relying on JSON mode, inject a tool that captures structured output.

## Model Abstraction
- 18+ providers via Vercel `ai` SDK.
- Model profiles with limits, capabilities, cost, variants.
- Middleware wrapping support.

## What Sophia Can Learn
1. **While loop, not graph**: OpenCode/Kilocode both prove that a simple loop with model-driven decisions works. LangGraph adds complexity without clear benefit for Sophia's use case.
2. **Structured output via tool**: When you need structured responses (like plan_turn, judge_evidence), define a tool for it rather than hoping for JSON in free text.
3. **Agent permissions as tool filters**: Different agents see different tools. Achievable without separate registries — just filter the tool list per agent.
4. **Compaction > truncation**: Summarizing old context is better than dropping it. Sophia currently has no context management at all (relies on LangGraph checkpoint + fixed evidence slice).
5. **Environment in system prompt**: Model name, working directory, date — simple but effective grounding.

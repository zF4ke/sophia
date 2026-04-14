# Kilocode — Architecture Study

## What It Is
Monorepo coding agent (fork/evolution of OpenCode). Built on **Effect.ts** for structured concurrency. Multi-agent system with explicit plan/build/explore modes.

## Agent Loop
- **Stream-based reactive loop** in `session/processor.ts`.
- Uses Vercel's `ai` SDK for LLM streaming with `streamText()`.
- Model streams events → processor handles each event type:
  - `tool-call` → execute tool, track doom loops
  - `tool-result` → capture output, truncate if needed
  - `text-start/delta/end` → stream text to UI
  - `finish` → mark complete
- After each LLM step: check overflow → optionally trigger compaction → continue or break.
- **Implicit state machine**: idle → busy → idle|retry|offline.
- Exit conditions: model finishes, no pending tool calls, max steps reached, or blocked by permissions.

**Key takeaway**: Still fundamentally a while-loop, but wrapped in Effect.ts reactive patterns. The model drives everything.

## Tool System
- ~17 built-in tools defined with **Zod schemas** for parameter validation.
- Tool interface: `{ description, parameters: ZodSchema, execute(args, ctx) }`.
- Execution returns: `{ title, metadata, output, attachments? }`.
- **Tool context** is rich: `{ sessionID, messageID, agent, abort, messages, metadata(), ask() }`.
- **Doom loop detection**: Same tool called 3x with identical input → permission prompt.
- **Output truncation**: 2000 lines OR 50KB, whichever first. Full output saved to disk with path hint.

**Key takeaway**: Zod validation + rich context per tool + doom-loop detection is a pattern worth adopting.

## Planning (Multi-Agent)
- **Explicit agent switching** rather than inline planning:
  - `build` (default) — full tool access
  - `plan` — read-only, writes plan files to `.opencode/plans/*.md`
  - `explore` — fast grep/glob/webfetch codebase search
  - `general` — multi-step research, no edits
  - `compaction` — internal, summarizes old messages
- Each agent has its own permissions, model, system prompt, temperature, and max steps.
- Plan mode: explore → design → review → write plan → exit via `plan_exit` tool.

**Key takeaway**: Agent specialization is powerful. Different permission sets and prompts for different tasks, rather than one monolithic agent.

## Context Management
- **Overflow detection**: `isOverflow()` checks token count against model limits minus a 20k buffer.
- **Two-phase compaction**:
  1. **Prune old tool outputs** (free tokens by erasing old tool output text, keep metadata).
  2. **Summarize** if still overflowing (invoke hidden "summary" agent to compress history).
- Tool output pruning is clever: keep the tool call metadata (what was called, when) but erase the verbose output text.
- Recent turns always protected from pruning.

**Key takeaway**: The prune-then-summarize strategy is smart. Sophia should prune evidence from old turns before paying for a summarization call.

## Error Handling
- **Retry policy**: 2s initial delay, 2x backoff, respects `retry-after` headers.
- **Retryable detection**: rate limits → retry; auth errors → don't retry; context overflow → don't retry.
- Error types: `APIError`, `ContextOverflowError`, `StructuredOutputError`, `AbortedError`, `AuthError`.
- **Session status tracking**: `idle`, `retry` (with countdown), `busy`, `offline`.
- Tool failures stored in message history for debugging.

**Key takeaway**: Clean error categorization. Sophia should distinguish "retryable" from "fatal" more explicitly.

## System Prompts
- **Model-specific base prompts**: `PROMPT_GPT`, `PROMPT_ANTHROPIC`, `PROMPT_GEMINI`, `PROMPT_BEAST`, etc.
- **Soul file** (`soul.txt`): core identity and behavior rules.
- **Composition order**: soul → agent-specific or model-specific → system extras → user-provided.
- **Skills injection**: available skills listed in system prompt.
- **Environment context**: cwd, platform, git status, editor context.
- **Plugin hooks** can transform system prompts at runtime.

**Key takeaway**: Model-specific prompts are important — different models need different instruction styles. Sophia currently uses one system prompt for all models.

## Model Abstraction
- 18+ providers via Vercel's `ai` SDK.
- Model registry with context limits, capabilities, cost, variants.
- Middleware pattern for model wrapping.
- Provider-specific error handling.

## What Sophia Can Learn
1. **Agent specialization**: Different agents for different tasks (explore vs build vs plan).
2. **Doom loop detection**: If the model calls the same thing 3x, intervene.
3. **Tool output pruning**: Free context space by erasing old tool output text while keeping metadata.
4. **Model-specific prompts**: Claude vs GPT need different instruction styles.
5. **Zod-based tool validation**: Catch bad arguments before execution.
6. **Rich tool context**: Give tools session context, abort signals, and ability to prompt user.

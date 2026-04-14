# Cline — Architecture Study

## What It Is
VS Code coding agent. Model-led iterative loop with tool execution, auto-approval, and checkpoint rollback.

## Agent Loop
- **Pattern**: Simple `while (!abort)` loop calling `recursivelyMakeClineRequests`.
- **No state machine**. No LangGraph. No explicit DAG. Just a loop.
- Model streams a response → tool calls are parsed → tool handlers execute → results appended as user messages → loop continues.
- Loop terminates when the model calls `attempt_completion` or user cancels.
- A `consecutiveMistakeCount` nudges the model when it outputs text without tool calls.

**Key takeaway**: The loop is dead simple. No planning DAG, no judge nodes, no classification. The model *is* the planner.

## Tool System
- **Static enum** of ~25 tools (`ClineDefaultTool`).
- **Handler registry** (`ToolExecutorCoordinator`): maps each tool name → a `IToolHandler` class.
- Handler classes are instantiated per-call with a `ToolValidator` for parameter checking.
- Tool results are fed back as `tool_result` content blocks (native Anthropic format).
- **Parallel tool calling** supported when model/provider allows it.
- **Auto-approval** per tool and per path. Human-in-the-loop by default.

**Key takeaway**: One handler class per tool. Clean separation. No argument enrichment — the model provides all arguments directly.

## Planning
- **No explicit planning phase.** The model reasons inline (via `<thinking>` tags for Claude).
- The system prompt gives the model full context (tools, rules, skills, environment) and lets it decide everything.
- Termination: model must call `attempt_completion` — there is no separate judge.

**Key takeaway**: Trust the model more. Don't add orchestration nodes to second-guess it.

## Context Management
- **Sliding window truncation**: `ContextManager` tracks a `conversationHistoryDeletedRange`.
- Truncation strategies: "quarter" (aggressive) and "standard".
- **PreCompact hook**: user-extensible compaction logic before aggressive truncation.
- Full history stored locally, truncated subset sent to API.
- Separate trackers: `FileContextTracker`, `ModelContextTracker`, `EnvironmentContextTracker`.

**Key takeaway**: Simple truncation, not summarization. Keep full history locally, send a window to the model.

## Error Handling
- Up to **3 auto-retries** with exponential backoff (2s, 4s, 8s).
- **Context window exceeded** → auto-truncate then retry (special path).
- Auth/spend-limit errors skip auto-retry.
- Tool failures → model gets the error and adapts.
- Checkpoint rollback available for tool execution disasters.

**Key takeaway**: Retry is mechanical — backoff + truncate. No sophisticated recovery logic needed.

## System Prompts
- `PromptRegistry` composes system prompt dynamically from:
  - Base role definition
  - `.clinerules` files (global + workspace)
  - Tool definitions (conditional on context)
  - Skills registry
  - Environment metadata (IDE, tabs, workspace)
- Prompts are **template-rendered at runtime**, not hardcoded strings.

**Key takeaway**: Modular composition. Similar to Sophia's `PromptRegistry` but richer context injection.

## Model Abstraction
- 30+ providers via adapter pattern: `createHandlerForProvider()` switch.
- Common `ApiHandler` interface: `createMessage()`, `getModel()`, `abort()`.
- Plan vs Act mode can use different models.
- Stream normalization layer across all providers.

## What Sophia Can Learn
1. **Simpler loop**: Cline proves a while-loop works fine. No graph needed.
2. **No judge node**: The model self-judges. Cline doesn't have a separate LLM call to evaluate evidence.
3. **No classification step**: No "conversation vs research" routing. The model decides what to do based on context.
4. **Flat tool dispatch**: One handler per tool, model provides all args. No argument enrichment layer.
5. **Checkpoint rollback**: Useful for recovery — save state before risky operations.

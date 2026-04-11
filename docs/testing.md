# Testing

Sophia uses two test lanes on purpose.

## Fast Deterministic Tests

Run:

```bash
npm run test
```

This is the default engineering suite and the one used by `npm run check`.

It keeps the model mocked so the runtime can be validated deterministically:
- capability wiring
- current-guild discovery
- retrieval contracts
- conversation continuity
- debug rendering
- storage behavior
- runtime guardrails

These tests are not trying to prove that a remote model will always choose the perfect plan. They are there to prove that the runtime, tool contracts, and orchestration stay correct.

## Live Model Tests

Run:

```bash
LIVE_MODEL_TESTS=1 OPENROUTER_API_KEY=... npm run test:live
```

The live suite is opt-in and expensive. It is excluded from `npm run check`.

It hits the real model API to validate:
- planner behavior with real prompts
- synthesis behavior with real prompts
- runtime stories where the model must stay conversational after tool use

The live suite still mocks non-model infrastructure when needed so the test isolates model behavior instead of depending on live Discord or production storage state.

## Why Both Lanes Exist

Using only mocked tests is not enough, because the model can still behave badly even when the runtime contract is correct.

Using only live tests is also not enough, because:
- they are expensive
- they are slower
- they are less deterministic
- they are a poor fit for low-level contract validation

The intended workflow is:
1. keep `npm run check` fast and strict
2. use `npm run test:live` before merging prompt/runtime changes that depend on real model behavior
3. promote important regressions from live findings into deterministic coverage when possible

## Current Live Coverage

The live suite currently checks:
- greeting planning stays conversational
- current-guild research planning is selected for explicit member/channel references
- exact member-id answers stay natural
- grounded retrieval answers stay natural after tool execution

Add new live stories when changing:
- planner prompts
- synthesis prompts
- tool-composition behavior
- answer style after grounded retrieval

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
- follow-up evidence reuse across turns
- continuation input gating (no automatic cross-turn exclusions without explicit continuation intent)
- strict scoped empty-result recovery retries in retrieval
- insufficient-confidence no-speculation guard behavior

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

Recent deterministic regression additions are in:
- `tests/runtime/runtimeUserStories.test.ts` (follow-up can answer from reused prior evidence without rerunning tools)
- `tests/runtime/planning.test.ts` (fresh follow-ups do not auto-apply cursor/exclusions; model tool choice is honored without pre-flight redirects; ambiguous members profiled before generic channel discovery)
- `tests/discord/UnifiedMessageRetrieval.test.ts` (strict scoped retry recovers hidden rows)
- `tests/discord/DiscordLiveService.test.ts` (list_members pagination with offset, no duplicate members across pages, default page size of 20, get_member_profile returns enriched fields: joinedAt, accountCreatedAt, premiumSince, pending)

## Manual Integration Tests

These are the canonical real-server test cases used to validate end-to-end behavior. They require a live bot instance and a real guild.

### The Two Glonos Problem

Tests ambiguous identity detection, profile escalation, and decision-quality synthesis.

**Prompt:**
> Olá, @Sophia. Eu estou meio confuso tem dois seres com o vulgo Glonos e agora que lascou de vez. Como eu posso saber qual é o verdadeiro? Eu preciso mandar uma mensagem muito importante para o verdadeiro, é uma questão de vida ou morte, então, por favor, não erre! @Sophia

**Expected behavior:**
- Fetches a profile for each member sharing the display name "Glonos" (not just one)
- Compares roles, join dates, and activity
- Returns a specific recommendation: which one is the real one and why
- Does not ask the user to choose manually or return a generic answer

**What this covers:**
- Ambiguous display name detection after `list_members` or `resolve_member_identity`
- `get_member_profile` called multiple times with different identifiers in one research pass
- Profile evidence includes join date and account age for disambiguation
- Synthesis makes a concrete decision instead of deflecting

---

### The Link of February 9th

Tests temporal scoping, multi-tool composition, and follow-up continuity.

**Prompt:**
> @Sophia 9 de fevereiro o openrosen mandou um link do youtube de uma musica para o canal de comandos. ele chegou a referir de quem era?

**Expected behavior:**
- Resolves `openrosen` as a member
- Resolves `comandos` as a channel
- Retrieves messages with author + channel + time bounds (after Feb 9, before Feb 10)
- Answers from message evidence: yes, and quotes what was said about the song

**Follow-up:**
> kkkkkkkk vdd eu lembro. ele diz que foi assistir bem "ecletico" sei la como escreve.

**Expected follow-up behavior:**
- Reuses the prior evidence (no retrieval rerun needed)
- Responds naturally to the conversational aside ("cetico") without going into research mode

**What this covers:**
- Date parsing from natural Portuguese
- Member + channel + time scope composition
- Evidence-based answer without hallucination
- Follow-up continuity without unnecessary tool re-execution

---

### The 4 Services Problem

Tests category discovery, structure inspection, scoped retrieval, and response quality under time pressure framing.

**Prompt:**
> @Sophia estou com pressa e precisava de um resumo do que tem nos canais do Serviços. quero uma descrição de cada serviço

**Expected behavior:**
- Resolves "Serviços" as a category
- Inspects the guild structure to enumerate child channels
- Retrieves scoped messages from each child channel
- Returns a brief description of each service based on message content, not invented summaries

**What this covers:**
- Category → guild structure → scoped retrieval chain
- Multiple `retrieve_messages` calls with different channel scopes
- Model decides the composition without being pre-wired to run the sequence
- Budget behavior under multi-channel retrieval
